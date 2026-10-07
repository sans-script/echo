use serde_json::{json, Value};
use std::{
    io::{BufRead, BufReader},
    sync::atomic::{AtomicBool, Ordering},
    time::{Duration, Instant},
};

#[derive(Debug)]
pub enum OllamaClientError {
    Transport(String),
    Http { status: u16, body: String },
    Malformed(String),
    Timeout(String),
    Cancelled,
}
impl std::fmt::Display for OllamaClientError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Transport(e) => write!(f, "Error communicating with Ollama: {e}"),
            Self::Http { status, body } => write!(f, "Ollama API returned HTTP {status}: {body}"),
            Self::Malformed(e) => write!(f, "Malformed response from Ollama: {e}"),
            Self::Timeout(e) => write!(f, "Request to Ollama timed out: {e}"),
            Self::Cancelled => write!(f, "Request cancelled."),
        }
    }
}
impl std::error::Error for OllamaClientError {}

#[derive(Debug, Clone)]
pub struct StreamEvent { pub kind: StreamEventKind }

#[derive(Debug, Clone)]
pub enum StreamEventKind {
    Content(String),
    ToolCallDelta(Value),
    Done {
        message: Value,
        elapsed: Duration,
        finish_reason: Option<String>,
        usage: Option<Value>,
    },
}

/// Per-request settings that only Ollama's native `/api/chat` accepts.
/// `num_ctx` must be identical across requests: a different value makes
/// Ollama reload the model and discard its prompt cache.
#[derive(Debug, Clone)]
pub struct ChatOptions {
    pub temperature: f64,
    pub num_ctx: u32,
    pub keep_alive: String,
}

#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InstalledModel {
    pub name: String,
    pub size: u64,
}

/// Ollama treats "name" and "name:latest" as the same model.
pub fn normalize_model_name(name: &str) -> String {
    if name.contains(':') { name.to_string() } else { format!("{name}:latest") }
}

#[derive(Debug, Clone)]
pub struct OllamaClient {
    pub base_url: String,
    pub timeout: Duration,
    http: reqwest::blocking::Client,
}impl OllamaClient {
    pub fn new(base_url: impl Into<String>, timeout: Duration) -> Result<Self, OllamaClientError> {
        let http = reqwest::blocking::Client::builder()
            .timeout(timeout).build()
            .map_err(|e| OllamaClientError::Transport(e.to_string()))?;
        Ok(Self {
            base_url: base_url.into().trim_end_matches('/').to_string(),
            timeout, http,
        })
    }

    pub fn check_health(&self, required_model: Option<&str>) -> Result<(), OllamaClientError> {
        let models = self.list_models(Duration::from_secs(10))?;
        if let Some(required) = required_model {
            let found = models.iter()
                .any(|m| m.name == required || m.name.starts_with(&format!("{required}:")));
            if !found {
                let names = models.iter().map(|m| m.name.as_str()).collect::<Vec<_>>();
                return Err(OllamaClientError::Malformed(
                    format!("Model '{required}' not found in available models: {names:?}")
                ));
            }
        }
        Ok(())
    }

    /// Installed models from `/api/tags`, sorted by name.
    pub fn list_models(&self, timeout: Duration) -> Result<Vec<InstalledModel>, OllamaClientError> {
        let response = self.http.get(format!("{}/api/tags", self.base_url))
            .timeout(timeout).send()
            .map_err(|e| if e.is_timeout() {
                OllamaClientError::Timeout(e.to_string())
            } else { OllamaClientError::Transport(e.to_string()) })?;
        let status = response.status();
        let body = response.text().unwrap_or_default();
        if !status.is_success() {
            return Err(OllamaClientError::Http { status: status.as_u16(), body });
        }
        let data: Value = serde_json::from_str(&body)
            .map_err(|e| OllamaClientError::Malformed(e.to_string()))?;
        let mut models = data.get("models").and_then(Value::as_array).into_iter().flatten()
            .filter_map(|m| {
                let name = m.get("name").or_else(|| m.get("model")).and_then(Value::as_str)?;
                (!name.is_empty()).then(|| InstalledModel {
                    name: name.to_string(),
                    size: m.get("size").and_then(Value::as_u64).unwrap_or(0),
                })
            })
            .collect::<Vec<_>>();
        models.sort_by_key(|m| m.name.to_lowercase());
        Ok(models)
    }

    /// Loads the model and evaluates the system prompt + tool definitions so
    /// the first real request only has to process the user's message.
    /// Ollama reuses the cached prompt prefix as long as it stays identical.
    pub fn warm_up(
        &self, model: &str, system_prompt: &str, tools: &[Value], options: &ChatOptions,
    ) -> Result<Duration, OllamaClientError> {
        let mut payload = request_payload(
            model, &[json!({"role": "system", "content": system_prompt})],
            Some(tools), options, false,
        );
        payload["options"]["num_predict"] = json!(1);
        let started = Instant::now();
        let response = self.post_chat(&payload)?;
        let _ = response.text();
        Ok(started.elapsed())
    }

    pub fn chat_completion_stream<F>(
        &self, model: &str, messages: &[Value], tools: Option<&[Value]>,
        options: &ChatOptions, cancel: &AtomicBool, mut on_event: F,
    ) -> Result<(Value, Duration), OllamaClientError>
    where F: FnMut(StreamEvent) {
        let payload = request_payload(model, messages, tools, options, true);
        let started = Instant::now();
        let response = self.post_chat(&payload)?;

        let mut content = String::new();
        let mut tool_calls = Vec::<Value>::new();
        let mut finish_reason = None;
        let mut usage = None;
        let mut role = "assistant".to_string();

        // Native streaming is newline-delimited JSON, one object per chunk.
        // Returning early drops the connection, which makes Ollama stop
        // generating.
        for line in BufReader::new(response).lines() {
            if cancel.load(Ordering::Relaxed) {
                return Err(OllamaClientError::Cancelled);
            }
            let line = line.map_err(|e| OllamaClientError::Transport(e.to_string()))?;
            if line.trim().is_empty() { continue; }
            let chunk: Value = match serde_json::from_str(&line) {
                Ok(v) => v, Err(_) => continue
            };
            if let Some(error) = chunk.get("error").and_then(Value::as_str) {
                return Err(OllamaClientError::Malformed(error.to_string()));
            }
            if let Some(message) = chunk.get("message") {
                if let Some(value) = message.get("role").and_then(Value::as_str) { role = value.to_string(); }
                if let Some(piece) = message.get("content").and_then(Value::as_str) {
                    if !piece.is_empty() {
                        content.push_str(piece);
                        on_event(StreamEvent { kind: StreamEventKind::Content(piece.to_string()) });
                    }
                }
                // Native tool calls arrive complete, with object arguments.
                // They are converted to the OpenAI shape the orchestrator uses.
                if let Some(calls) = message.get("tool_calls").and_then(Value::as_array) {
                    let converted = calls.iter().map(|call| {
                        let function = call.get("function").cloned().unwrap_or_else(|| json!({}));
                        let arguments = match function.get("arguments") {
                            Some(Value::String(s)) => s.clone(),
                            Some(other) => other.to_string(),
                            None => "{}".to_string(),
                        };
                        let index = tool_calls.len();
                        let id = call.get("id").and_then(Value::as_str)
                            .map(str::to_string).unwrap_or_else(|| format!("call_{index}"));
                        let converted = json!({
                            "id": id, "type": "function",
                            "function": {
                                "name": function.get("name").and_then(Value::as_str).unwrap_or(""),
                                "arguments": arguments,
                            }
                        });
                        tool_calls.push(converted.clone());
                        converted
                    }).collect::<Vec<_>>();
                    on_event(StreamEvent {
                        kind: StreamEventKind::ToolCallDelta(Value::Array(converted)),
                    });
                }
            }
            if chunk.get("done").and_then(Value::as_bool).unwrap_or(false) {
                finish_reason = chunk.get("done_reason").and_then(Value::as_str).map(str::to_string);
                usage = Some(json!({
                    "prompt_eval_count": chunk.get("prompt_eval_count"),
                    "prompt_eval_duration": chunk.get("prompt_eval_duration"),
                    "eval_count": chunk.get("eval_count"),
                    "eval_duration": chunk.get("eval_duration"),
                    "load_duration": chunk.get("load_duration"),
                    "total_duration": chunk.get("total_duration"),
                }));
                break;
            }
        }

        let mut message = json!({"role": role, "content": content});
        if !tool_calls.is_empty() {
            message["tool_calls"] = Value::Array(tool_calls);
        }
        let elapsed = started.elapsed();
        on_event(StreamEvent {
            kind: StreamEventKind::Done {
                message: message.clone(), elapsed, finish_reason, usage
            },
        });
        Ok((message, elapsed))
    }

    fn post_chat(&self, payload: &Value) -> Result<reqwest::blocking::Response, OllamaClientError> {
        let response = self.http
            .post(format!("{}/api/chat", self.base_url))
            .json(payload).send()
            .map_err(|e| if e.is_timeout() {
                OllamaClientError::Timeout(format!("after {:?}", self.timeout))
            } else { OllamaClientError::Transport(e.to_string()) })?;

        let status = response.status();
        if !status.is_success() {
            return Err(OllamaClientError::Http {
                status: status.as_u16(), body: response.text().unwrap_or_default()
            });
        }
        Ok(response)
    }
}

fn request_payload(
    model: &str, messages: &[Value], tools: Option<&[Value]>, options: &ChatOptions, stream: bool,
) -> Value {
    let mut payload = json!({
        "model": model,
        "messages": messages.iter().map(to_native_message).collect::<Vec<_>>(),
        "stream": stream,
        "keep_alive": options.keep_alive,
        "options": {"temperature": options.temperature, "num_ctx": options.num_ctx},
    });
    if let Some(tools) = tools {
        payload["tools"] = Value::Array(tools.to_vec());
    }
    payload
}

/// Converts an OpenAI-style history message to the shape `/api/chat` expects:
/// tool call arguments as JSON objects and tool results tagged with
/// `tool_name`.
fn to_native_message(message: &Value) -> Value {
    let mut native = message.clone();
    if let Some(calls) = message.get("tool_calls").and_then(Value::as_array) {
        native["tool_calls"] = calls.iter().map(|call| {
            let function = call.get("function").cloned().unwrap_or_else(|| json!({}));
            let arguments = match function.get("arguments") {
                Some(Value::String(s)) => serde_json::from_str::<Value>(s)
                    .ok().filter(Value::is_object).unwrap_or_else(|| json!({})),
                Some(other) => other.clone(),
                None => json!({}),
            };
            json!({"function": {"name": function.get("name").cloned().unwrap_or_default(), "arguments": arguments}})
        }).collect();
    }
    if message.get("role").and_then(Value::as_str) == Some("tool") {
        if let Some(name) = message.get("name") {
            native["tool_name"] = name.clone();
        }
    }
    native
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn normalizes_url() {
        let c = OllamaClient::new("http://localhost:11434///", Duration::from_secs(1)).unwrap();
        assert_eq!(c.base_url, "http://localhost:11434");
    }

    #[test]
    fn converts_history_to_native_shape() {
        let assistant = to_native_message(&json!({
            "role": "assistant", "content": "",
            "tool_calls": [{"id": "call_0", "type": "function",
                "function": {"name": "read_file", "arguments": "{\"path\":\"a.txt\"}"}}]
        }));
        assert_eq!(assistant["tool_calls"][0]["function"]["arguments"]["path"], "a.txt");

        let tool = to_native_message(&json!({
            "role": "tool", "tool_call_id": "call_0", "name": "read_file", "content": "hi"
        }));
        assert_eq!(tool["tool_name"], "read_file");
    }

    #[test]
    fn payload_carries_native_options() {
        let options = ChatOptions { temperature: 0.0, num_ctx: 8192, keep_alive: "30m".into() };
        let payload = request_payload("m", &[], None, &options, true);
        assert_eq!(payload["options"]["num_ctx"], 8192);
        assert_eq!(payload["keep_alive"], "30m");
        assert!(payload.get("tools").is_none());
    }
}
