use crate::{
    config::EchoConfig,
    ollama::{ChatOptions, OllamaClient, OllamaClientError, StreamEventKind},
    tools::registry::{ToolExecutionRecord, ToolRegistry},
    workspace_helpers::tree_workspace,
};
use serde_json::{json, Map, Value};
use std::{
    sync::{Arc, atomic::{AtomicBool, Ordering}},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

const RESPONSE_RULES: &str = "
<response_rules>
Use environment information internally.
Do not copy raw environment values into normal user-facing answers.
For date/time questions, convert the ISO timestamp to natural human-readable
language. Do not include the raw timestamp unless the user asks for it.
For environment questions, answer conversationally instead of reproducing
the environment fields.
</response_rules>
";
const LISTING_NOTE: &str = "\n[Note: this listing is already displayed to the user. Do not repeat it; answer with one short sentence.]";
const DECLINED_RESULT: &str = "The user declined this change in the confirmation prompt. The tool was not executed and nothing was modified. This is not an error: briefly confirm to the user that the change was cancelled as they chose, in the same language as their request.";
const REPEAT_NOTICE: &str = "\n[Notice: You have already executed this tool with the same arguments. Please synthesize your final response.]";
const WORKSPACE_SNAPSHOT_LINES: usize = 60;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunStopReason { Completed, Error, UserCancelled, MaxIterations }

impl std::fmt::Display for RunStopReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Completed => "completed",
            Self::Error => "error",
            Self::UserCancelled => "user_cancelled",
            Self::MaxIterations => "max_iterations_reached",
        })
    }
}

#[allow(dead_code)]
#[derive(Debug)]
pub struct OrchestratorResult {
    pub user_prompt: String,
    pub final_response: String,
    pub iterations: usize,
    pub tool_executions: Vec<ToolExecutionRecord>,
    pub messages: Vec<Value>,
    pub total_duration: Duration,
    pub model_name: String,
    pub stopped_reason: RunStopReason,
    pub error_message: Option<String>,
}

#[derive(Debug)]
pub enum OrchestratorEvent {
    IterationStarted,
    ContentDelta(String),
    ToolCallReceived { tool_name: String, arguments: String },
    ToolExecuted(ToolExecutionRecord),
    EmptyResponseRetry,
}

pub struct EchoOrchestrator {
    pub config: EchoConfig,
    pub client: OllamaClient,
    pub registry: ToolRegistry,
    pub history: Vec<Value>,
    /// Set from another thread to stop the current run at the next chunk.
    pub cancel: Arc<AtomicBool>,
    /// True when the UI already shows tool results to the user, so listings
    /// don't need to be repeated by the model.
    pub tool_output_visible: bool,
    /// System prompt for the current conversation. It is built once (with a
    /// workspace snapshot) and then kept byte-identical so Ollama can reuse
    /// its cached evaluation on every turn.
    session_prompt: Option<String>,
}

impl EchoOrchestrator {
    pub fn new(config: EchoConfig, registry: ToolRegistry) -> Result<Self, OllamaClientError> {
        let client = OllamaClient::new(&config.ollama_url, Duration::from_secs_f64(config.request_timeout))?;
        Ok(Self {
            config, client, registry, history: Vec::new(),
            cancel: Arc::new(AtomicBool::new(false)),
            tool_output_visible: true,
            session_prompt: None,
        })
    }

    pub fn reset_history(&mut self) {
        self.history.clear();
        self.session_prompt = None;
    }

    fn chat_options(&self) -> ChatOptions {
        ChatOptions {
            temperature: self.config.temperature,
            num_ctx: self.config.num_ctx,
            keep_alive: self.config.keep_alive.clone(),
        }
    }

    fn system_prompt(&mut self) -> String {
        if let Some(prompt) = &self.session_prompt {
            return prompt.clone();
        }
        let snapshot = tree_workspace(&self.config.workspace, 2)
            .lines().take(WORKSPACE_SNAPSHOT_LINES).collect::<Vec<_>>().join("\n");
        let prompt = format!(
            "{}\n\nWorkspace root: {}\nCurrent workspace snapshot (use these exact paths and do not guess file names):\n{}\n{}",
            self.config.system_prompt, self.config.workspace.display(), snapshot, RESPONSE_RULES
        );
        self.session_prompt = Some(prompt.clone());
        prompt
    }

    /// Date, time and machine details for the current turn. They go into the
    /// user message rather than the system prompt: a timestamp in the system
    /// prompt would change on every turn and invalidate Ollama's prompt cache.
    fn environment_context(&self) -> String {
        let now = chrono::Local::now();
        // The shell run_command uses, so the model writes matching commands.
        let shell = crate::tools::shell::shell_name();
        format!(
            "<environment>\nCurrent date/time: {}\nTimezone: UTC{}\nOperating system: {}\nShell: {}\nWorking directory: {}\nModel: {}\n</environment>",
            now.to_rfc3339_opts(chrono::SecondsFormat::Secs, false),
            now.format("%:z"), std::env::consts::OS, shell,
            self.config.workspace.display(), self.config.model,
        )
    }

    /// Returns a job that preloads the model with the same system prompt,
    /// tools and options `run` uses, so it can run on a background thread.
    pub fn warm_up_task(&mut self) -> impl FnOnce() -> Result<Duration, OllamaClientError> + Send + 'static {
        let client = self.client.clone();
        let model = self.config.model.clone();
        let system_prompt = self.system_prompt();
        let tools = self.registry.definitions();
        let options = self.chat_options();
        move || client.warm_up(&model, &system_prompt, &tools, &options)
    }

    /// Runs one turn. `confirm` is asked before executing any tool that
    /// requires confirmation; returning false ends the turn without changes.
    pub fn run<F, C>(&mut self, user_prompt: impl Into<String>, system_prompt: Option<&str>, mut on_event: F, mut confirm: C) -> OrchestratorResult
    where F: FnMut(OrchestratorEvent), C: FnMut(&str, &Value) -> bool {
        let user_prompt = user_prompt.into();
        let started = std::time::Instant::now();
        self.cancel.store(false, Ordering::Relaxed);
        let options = self.chat_options();
        let previous_len = self.history.len();

        let system = match system_prompt {
            Some(prompt) => prompt.to_string(),
            None => self.system_prompt(),
        };
        let mut messages = vec![json!({"role": "system", "content": system})];
        messages.extend(self.history.clone());
        // The environment goes in its own message so the request itself
        // stays the last, unmixed text the model reads (a small model tends
        // to answer in English when the request is wrapped in English).
        messages.push(json!({"role": "user", "content": self.environment_context()}));
        messages.push(json!({"role": "user", "content": user_prompt}));

        let tools = self.registry.definitions();
        let mut executions = Vec::new();
        let mut iterations = 0;
        let mut empty_retries = 0;
        // After a repeated or declined call the model must answer in text.
        // Tools stay in the request so Ollama can reuse its cached prompt;
        // they are only dropped if the model insists on calling one anyway.
        let mut force_text = false;
        let mut strip_tools = false;
        let mut last_signature: Option<String> = None;

        while iterations < self.config.max_iterations {
            iterations += 1;
            on_event(OrchestratorEvent::IterationStarted);

            // Stream text right away, but hold back content that may be a
            // tool call written as JSON until the response is complete.
            let mut held = String::new();
            let mut streaming = false;
            let result = self.client.chat_completion_stream(
                &self.config.model, &messages,
                if strip_tools { None } else { Some(&tools) },
                &options, &self.cancel,
                |event| if let StreamEventKind::Content(delta) = event.kind {
                    if streaming {
                        on_event(OrchestratorEvent::ContentDelta(delta));
                        return;
                    }
                    held.push_str(&delta);
                    let probe = held.trim_start();
                    let maybe_tool = probe.is_empty() || probe.starts_with('{') || "```json".starts_with(probe) || probe.starts_with("```json");
                    if !maybe_tool {
                        on_event(OrchestratorEvent::ContentDelta(std::mem::take(&mut held)));
                        streaming = true;
                    }
                }
            );

            let (mut message, _) = match result {
                Ok(value) => value,
                Err(OllamaClientError::Cancelled) => return self.cancelled_result(user_prompt, messages, executions, iterations, started.elapsed()),
                Err(error) => return self.error_result(user_prompt, messages, executions, iterations, started.elapsed(), error)
            };

            let empty = message.get("content").and_then(Value::as_str).unwrap_or("").trim().is_empty();
            if empty && message.get("tool_calls").is_none() && !force_text && empty_retries < 1 {
                empty_retries += 1;
                on_event(OrchestratorEvent::EmptyResponseRetry);
                continue;
            }

            let mut calls = message.get("tool_calls").and_then(Value::as_array).cloned().unwrap_or_default();

            if calls.is_empty() {
                if let Some(call) = parse_fallback_tool_call(message.get("content").and_then(Value::as_str).unwrap_or("")) {
                    calls.push(call);
                    message["content"] = Value::String(String::new());
                }
            }
            if force_text && !calls.is_empty() {
                // Asked to answer, the model tried another tool: retry once
                // without tools, then give up on its calls.
                if !strip_tools {
                    strip_tools = true;
                    continue;
                }
                calls.clear();
            }
            let has_text = !message.get("content").and_then(Value::as_str).unwrap_or("").trim().is_empty();
            if calls.is_empty() && has_text && !held.trim().is_empty() {
                on_event(OrchestratorEvent::ContentDelta(held));
            }

            messages.push(message.clone());

            if calls.is_empty() {
                let text = message.get("content").and_then(Value::as_str).unwrap_or("").trim();
                let final_response = if text.is_empty() {
                    // Ollama drops tool calls it cannot parse (often a write
                    // with malformed JSON), which leaves an empty reply.
                    "The model returned an empty reply (it may have produced a tool call Ollama could not parse). Please try again or rephrase.".to_string()
                } else { text.to_string() };
                self.history = compact_history(
                    messages.iter().skip(1).cloned().collect(), previous_len,
                    self.config.history_tool_output_chars, self.config.num_ctx as usize * 2,
                );
                return OrchestratorResult {
                    user_prompt, final_response, iterations,
                    tool_executions: executions, messages,
                    total_duration: started.elapsed(),
                    model_name: self.config.model.clone(),
                    stopped_reason: RunStopReason::Completed,
                    error_message: None,
                };
            }

            let mut repeated = false;
            for call in calls {
                if self.cancel.load(Ordering::Relaxed) {
                    return self.cancelled_result(user_prompt, messages, executions, iterations, started.elapsed());
                }
                let call_id = call.get("id").and_then(Value::as_str).map(str::to_string)
                    .unwrap_or_else(|| format!("call_{}", unique_id()));
                let function = call.get("function").cloned().unwrap_or_else(|| json!({}));
                let name = function.get("name").and_then(Value::as_str).unwrap_or("").to_string();
                let raw = function.get("arguments").and_then(Value::as_str).unwrap_or("{}").to_string();

                let signature = call_signature(&name, &raw);
                let repeat = last_signature.as_deref() == Some(signature.as_str());
                repeated |= repeat;
                last_signature = Some(signature);

                on_event(OrchestratorEvent::ToolCallReceived {
                    tool_name: name.clone(), arguments: raw.clone()
                });

                if self.registry.requires_confirmation(&name) {
                    let args = serde_json::from_str::<Value>(&raw).ok().filter(Value::is_object)
                        .unwrap_or_else(|| json!({"_raw_arguments": raw}));
                    if !confirm(&name, &args) {
                        // The model explains the cancellation in its own words,
                        // in text only, so it can't retry or claim success.
                        executions.push(denied_execution(&call_id, &name, &raw, DECLINED_RESULT, args.as_object().cloned().unwrap_or_default()));
                        messages.push(json!({"role":"tool", "tool_call_id":call_id, "name":name, "content":DECLINED_RESULT}));
                        force_text = true;
                        break;
                    }
                }

                let execution = self.registry.execute(&call_id, &name, &raw);
                on_event(OrchestratorEvent::ToolExecuted(execution.clone()));
                let mut content = execution.result.clone();
                if self.tool_output_visible && execution.success && matches!(name.as_str(), "tree" | "list_directory") {
                    content.push_str(LISTING_NOTE);
                }
                if repeat {
                    content.push_str(REPEAT_NOTICE);
                }
                executions.push(execution);
                messages.push(json!({
                    "role":"tool", "tool_call_id":call_id, "name":name, "content":content
                }));
            }
            if repeated {
                force_text = true;
            }
        }

        OrchestratorResult {
            user_prompt,
            final_response: "Maximum tool-calling iterations reached before completing.".into(),
            iterations, tool_executions: executions, messages,
            total_duration: started.elapsed(), model_name: self.config.model.clone(),
            stopped_reason: RunStopReason::MaxIterations, error_message: None,
        }
    }

    fn cancelled_result(&self, user_prompt:String, messages:Vec<Value>, executions:Vec<ToolExecutionRecord>,
        iterations:usize, duration:Duration) -> OrchestratorResult {
        OrchestratorResult {
            user_prompt, final_response:"[Interrupted]".into(), iterations,
            tool_executions:executions, messages, total_duration:duration,
            model_name:self.config.model.clone(), stopped_reason:RunStopReason::UserCancelled,
            error_message:None,
        }
    }

    fn error_result(&self, user_prompt:String, messages:Vec<Value>, executions:Vec<ToolExecutionRecord>,
        iterations:usize, duration:Duration, error:OllamaClientError) -> OrchestratorResult {
        OrchestratorResult {
            user_prompt, final_response:String::new(), iterations,
            tool_executions:executions, messages, total_duration:duration,
            model_name:self.config.model.clone(), stopped_reason:RunStopReason::Error,
            error_message:Some(error.to_string()),
        }
    }
}

/// One-line execution summary shown after each turn and by `/stats`.
pub fn format_stats(result: &OrchestratorResult) -> String {
    let tools = result.tool_executions.len();
    // fold from +0.0: an empty f64 `sum` is -0.0 and would print "-0.00s".
    let tool_time = result.tool_executions.iter().fold(0.0, |total, e| total + e.duration_seconds);
    format!(
        "{} {} | {} {} | tool time {:.2}s | response {} chars | total {:.2}s | {}",
        result.iterations, if result.iterations == 1 { "iteration" } else { "iterations" },
        tools, if tools == 1 { "tool" } else { "tools" },
        tool_time, result.final_response.chars().count(),
        result.total_duration.as_secs_f64(), result.stopped_reason,
    )
}

/// Keeps the conversation inside the context window. Tool outputs from the
/// turn that just finished (messages from `turn_start` on) are cut to
/// `tool_output_chars`; earlier messages are left untouched so Ollama can keep
/// reusing its cached prompt prefix. Whole turns are dropped from the front
/// only when the history exceeds `max_chars`.
fn compact_history(mut history:Vec<Value>, turn_start:usize, tool_output_chars:usize, max_chars:usize)->Vec<Value> {
    for message in history.iter_mut().skip(turn_start) {
        if message.get("role").and_then(Value::as_str)!=Some("tool") { continue; }
        let Some(content)=message.get("content").and_then(Value::as_str) else { continue; };
        if content.len()<=tool_output_chars { continue; }
        let cut=floor_char_boundary(content,tool_output_chars);
        let total=content.len();
        message["content"]=Value::String(format!(
            "{}\n... [Older tool output cut to {cut} of {total} characters; call the tool again if you need the rest]",
            &content[..cut]
        ));
    }

    let size=|h:&[Value]| h.iter().map(|m| m.to_string().len()).sum::<usize>();
    let is_user=|m:&Value| m.get("role").and_then(Value::as_str)==Some("user");
    let is_environment=|m:&Value| is_user(m)
        && m.get("content").and_then(Value::as_str).is_some_and(|c| c.starts_with("<environment>"));
    while size(&history)>max_chars {
        // A turn starts at a user message (or at the environment message
        // that precedes it); never leave orphaned tool results.
        let next_turn=(1..history.len())
            .find(|&i| is_user(&history[i]) && !is_environment(&history[i-1]));
        match next_turn {
            Some(i)=>{ history.drain(..i); }
            None=>break,
        }
    }
    history
}

pub fn floor_char_boundary(text:&str,index:usize)->usize {
    let mut index=index.min(text.len());
    while !text.is_char_boundary(index) { index-=1; }
    index
}

/// Identifies a call for repetition detection. Listing the workspace root is
/// the same call whether the model passes ".", "" or nothing.
fn call_signature(name:&str, raw:&str)->String {
    let args=match serde_json::from_str::<Value>(raw) {
        Ok(mut value) => {
            if name=="list_directory" && matches!(value.get("path").and_then(Value::as_str), None|Some("")|Some(".")) {
                value=json!({});
            }
            value.to_string()
        }
        Err(_) => raw.trim().to_string(),
    };
    format!("{name}:{args}")
}

/// Some models write the tool call as JSON text (optionally fenced) instead
/// of using native tool calling.
fn parse_fallback_tool_call(content:&str)->Option<Value> {
    let mut text=content.trim();
    if text.starts_with("```") && text.ends_with("```") && text.len()>=6 {
        text=text.trim_matches('`').trim();
        if text.len()>=4 && text[..4].eq_ignore_ascii_case("json") { text=text[4..].trim(); }
    }
    if !(text.starts_with('{') && text.ends_with('}')) { return None; }
    let value:Value=serde_json::from_str(text).ok()?;
    let object=value.as_object()?;
    let name=object.get("name").or_else(||object.get("tool")).or_else(||object.get("tool_name"))
        .and_then(Value::as_str).filter(|name| !name.is_empty())?;
    let args=object.get("arguments").or_else(||object.get("args")).or_else(||object.get("parameters"))
        .cloned().unwrap_or_else(||json!({}));
    if !args.is_object(){return None;}
    Some(json!({"id":format!("call_{}",unique_id()),"type":"function",
        "function":{"name":name,"arguments":args.to_string()}}))
}

fn denied_execution(call_id:&str,tool_name:&str,raw:&str,result:&str,arguments:Map<String,Value>)->ToolExecutionRecord {
    ToolExecutionRecord {
        call_id:call_id.to_string(), tool_name:tool_name.to_string(), arguments,
        raw_arguments:raw.to_string(), result:result.to_string(),
        success:false, duration_seconds:0.0,
        error:Some("User denied confirmation.".into()),
    }
}

fn unique_id()->u128 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compaction_cuts_only_the_latest_turn_tool_outputs() {
        let old_tool=json!({"role":"tool","name":"read_file","content":"a".repeat(50)});
        let history=vec![
            json!({"role":"user","content":"first"}), old_tool.clone(),
            json!({"role":"user","content":"second"}),
            json!({"role":"tool","name":"read_file","content":"é".repeat(50)}),
        ];
        let compacted=compact_history(history,2,15,usize::MAX);
        assert_eq!(compacted[1],old_tool);
        let cut=compacted[3]["content"].as_str().unwrap();
        assert!(cut.starts_with(&"é".repeat(7)));
        assert!(cut.contains("cut to 14 of 100"));
    }

    #[test]
    fn compaction_drops_whole_oldest_turns() {
        let history=vec![
            json!({"role":"user","content":"first"}),
            json!({"role":"tool","name":"read_file","content":"x".repeat(200)}),
            json!({"role":"assistant","content":"done"}),
            json!({"role":"user","content":"second"}),
            json!({"role":"assistant","content":"ok"}),
        ];
        let compacted=compact_history(history,5,1000,150);
        assert_eq!(compacted.len(),2);
        assert_eq!(compacted[0]["content"],"second");
    }

    #[test]
    fn compaction_keeps_environment_with_its_request() {
        let history=vec![
            json!({"role":"user","content":"<environment>\nold\n</environment>"}),
            json!({"role":"user","content":"x".repeat(200)}),
            json!({"role":"assistant","content":"done"}),
            json!({"role":"user","content":"<environment>\nnew\n</environment>"}),
            json!({"role":"user","content":"second"}),
            json!({"role":"assistant","content":"ok"}),
        ];
        let compacted=compact_history(history,6,1000,200);
        assert_eq!(compacted.len(),3);
        assert!(compacted[0]["content"].as_str().unwrap().contains("new"));
        assert_eq!(compacted[1]["content"],"second");
    }

    #[test]
    fn fallback_parses_json_tool_call() {
        let c=parse_fallback_tool_call(r#"{"name":"read_file","arguments":{"path":"src/main.rs"}}"#).unwrap();
        assert_eq!(c["function"]["name"],"read_file");
        assert_eq!(c["function"]["arguments"],r#"{"path":"src/main.rs"}"#);
    }

    #[test]
    fn fallback_unwraps_fenced_json() {
        let c=parse_fallback_tool_call("```json\n{\"name\":\"tree\",\"arguments\":{}}\n```").unwrap();
        assert_eq!(c["function"]["name"],"tree");
        assert!(parse_fallback_tool_call("{\"not\":\"a call\"}").is_none());
    }

    #[test]
    fn listing_the_root_has_one_signature() {
        assert_eq!(call_signature("list_directory","{}"),call_signature("list_directory",r#"{"path":"."}"#));
        assert_ne!(call_signature("list_directory","{}"),call_signature("list_directory",r#"{"path":"src"}"#));
    }

    #[test]
    fn stats_line_matches_python_format() {
        let result=OrchestratorResult {
            user_prompt:"hi".into(), final_response:"Hello".into(), iterations:1,
            tool_executions:Vec::new(), messages:Vec::new(), total_duration:Duration::from_millis(2950),
            model_name:"m".into(), stopped_reason:RunStopReason::Completed, error_message:None,
        };
        assert_eq!(format_stats(&result),"1 iteration | 0 tools | tool time 0.00s | response 5 chars | total 2.95s | completed");
    }
}
