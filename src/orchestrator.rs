use crate::{
    config::EchoConfig,
    ollama::{OllamaClient, OllamaClientError, StreamEventKind},
    tools::registry::{ToolExecutionRecord, ToolRegistry},
};
use serde_json::{json, Map, Value};
use std::{collections::HashSet, time::{Duration, SystemTime, UNIX_EPOCH}};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunStopReason { Completed, Error, UserCancelled, MaxIterations }

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
    ContentDelta(String),
    ToolCallDelta(Value),
    ToolCallReceived { call_id: String, tool_name: String, arguments: String },
    ToolExecuted(ToolExecutionRecord),
    EmptyResponseRetry { retry: usize },
}

pub struct EchoOrchestrator {
    pub config: EchoConfig,
    pub client: OllamaClient,
    pub registry: ToolRegistry,
    pub history: Vec<Value>,
}

impl EchoOrchestrator {
    pub fn new(config: EchoConfig, registry: ToolRegistry) -> Result<Self, OllamaClientError> {
        let client = OllamaClient::new(&config.ollama_url, Duration::from_secs_f64(config.request_timeout))?;
        Ok(Self { config, client, registry, history: Vec::new() })
    }

    pub fn reset_history(&mut self) { self.history.clear(); }

    pub fn run<F>(&mut self, user_prompt: impl Into<String>, system_prompt: Option<&str>, mut on_event: F) -> OrchestratorResult
    where F: FnMut(OrchestratorEvent) {
        let user_prompt = user_prompt.into();
        let started = std::time::Instant::now();
        let mut messages = vec![json!({
            "role": "system",
            "content": system_prompt.unwrap_or(&self.config.system_prompt)
        })];
        messages.extend(self.history.clone());
        messages.push(json!({"role":"user","content":user_prompt}));

        let tools = self.registry.definitions();
        let mut executions = Vec::new();
        let mut iterations = 0;
        let mut empty_retries = 0;
        let mut force_text = false;
        let mut signatures = HashSet::new();

        while iterations < self.config.max_iterations {
            iterations += 1;
            let result = self.client.chat_completion_stream(
                &self.config.model, &messages,
                if force_text { None } else { Some(&tools) },
                self.config.temperature,
                |event| match event.kind {
                    StreamEventKind::Content(delta) => on_event(OrchestratorEvent::ContentDelta(delta)),
                    StreamEventKind::ToolCallDelta(delta) => on_event(OrchestratorEvent::ToolCallDelta(delta)),
                    StreamEventKind::Done { .. } => {}
                }
            );

            let (mut message, _) = match result {
                Ok(value) => value,
                Err(error) => return self.error_result(user_prompt, messages, executions, iterations, started.elapsed(), error)
            };

            let empty = message.get("content").and_then(Value::as_str).unwrap_or("").trim().is_empty();
            if empty && message.get("tool_calls").is_none() && !force_text && empty_retries < 1 {
                empty_retries += 1;
                on_event(OrchestratorEvent::EmptyResponseRetry { retry: empty_retries });
                continue;
            }

            let mut calls = message.get("tool_calls").and_then(Value::as_array).cloned().unwrap_or_default();

            if calls.is_empty() && !force_text {
                if let Some(call) = parse_fallback_tool_call(message.get("content").and_then(Value::as_str).unwrap_or("")) {
                    calls.push(call);
                    message["content"] = Value::String(String::new());
                }
            }

            messages.push(message.clone());

            if calls.is_empty() || force_text {
                let text = message.get("content").and_then(Value::as_str).unwrap_or("").trim();
                let final_response = if text.is_empty() {
                    "O modelo não retornou uma resposta. Tente novamente.".to_string()
                } else { text.to_string() };
                self.history = messages.iter().skip(1).cloned().collect();
                return OrchestratorResult {
                    user_prompt, final_response, iterations,
                    tool_executions: executions, messages,
                    total_duration: started.elapsed(),
                    model_name: self.config.model.clone(),
                    stopped_reason: RunStopReason::Completed,
                    error_message: None,
                };
            }

            for call in calls {
                let call_id = call.get("id").and_then(Value::as_str).map(str::to_string)
                    .unwrap_or_else(|| format!("call_{}", unique_id()));
                let function = call.get("function").cloned().unwrap_or_else(|| json!({}));
                let name = function.get("name").and_then(Value::as_str).unwrap_or("").to_string();
                let raw = function.get("arguments").and_then(Value::as_str).unwrap_or("{}").to_string();

                on_event(OrchestratorEvent::ToolCallReceived {
                    call_id: call_id.clone(), tool_name: name.clone(), arguments: raw.clone()
                });

                if !signatures.insert(format!("{}:{}", name, normalize_json(&raw))) {
                    force_text = true;
                }

                if self.registry.requires_confirmation(&name) {
                    let args = serde_json::from_str::<Value>(&raw).ok()
                        .and_then(|v| v.as_object().cloned()).unwrap_or_default();
                    let execution = cancelled_execution(&call_id, &name, args);
                    on_event(OrchestratorEvent::ToolExecuted(execution.clone()));
                    executions.push(execution);
                    return OrchestratorResult {
                        user_prompt,
                        final_response: format!("A operação '{name}' requer confirmação antes de modificar arquivos."),
                        iterations, tool_executions: executions, messages,
                        total_duration: started.elapsed(), model_name: self.config.model.clone(),
                        stopped_reason: RunStopReason::UserCancelled, error_message: None,
                    };
                }

                let execution = self.registry.execute(&call_id, &name, &raw);
                on_event(OrchestratorEvent::ToolExecuted(execution.clone()));
                let content = execution.result.clone();
                executions.push(execution);
                messages.push(json!({
                    "role":"tool", "tool_call_id":call_id, "name":name, "content":content
                }));
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

fn normalize_json(raw:&str)->String {
    serde_json::from_str::<Value>(raw).map(|v|v.to_string()).unwrap_or_else(|_|raw.trim().to_string())
}

fn parse_fallback_tool_call(content:&str)->Option<Value> {
    let text=content.trim();
    let value:Value=serde_json::from_str(text).ok()?;
    let object=value.as_object()?;
    let name=object.get("name").or_else(||object.get("tool")).or_else(||object.get("tool_name"))
        .and_then(Value::as_str)?;
    let args=object.get("arguments").or_else(||object.get("args")).or_else(||object.get("parameters"))
        .cloned().unwrap_or_else(||json!({}));
    if !args.is_object(){return None;}
    Some(json!({"id":format!("call_{}",unique_id()),"type":"function",
        "function":{"name":name,"arguments":args.to_string()}}))
}

fn cancelled_execution(call_id:&str,tool_name:&str,arguments:Map<String,Value>)->ToolExecutionRecord {
    ToolExecutionRecord {
        call_id:call_id.to_string(), tool_name:tool_name.to_string(), arguments,
        raw_arguments:String::new(),
        result:format!("Tool '{tool_name}' execution was not confirmed."),
        success:false, duration_seconds:0.0,
        error:Some("Confirmation is required.".into()),
    }
}

fn unique_id()->u128 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fallback_parses_json_tool_call() {
        let c=parse_fallback_tool_call(r#"{"name":"read_file","arguments":{"path":"src/main.rs"}}"#).unwrap();
        assert_eq!(c["function"]["name"],"read_file");
        assert_eq!(c["function"]["arguments"],r#"{"path":"src/main.rs"}"#);
    }
}