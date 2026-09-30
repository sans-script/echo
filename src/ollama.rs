use serde_json::{json, Value};
use std::{io::{BufRead, BufReader}, time::{Duration, Instant}};

#[derive(Debug)]
pub enum OllamaClientError {
    Transport(String),
    Http { status: u16, body: String },
    Malformed(String),
    Timeout(String),
}
impl std::fmt::Display for OllamaClientError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Transport(e) => write!(f, "Error communicating with Ollama: {e}"),
            Self::Http { status, body } => write!(f, "Ollama API returned HTTP {status}: {body}"),
            Self::Malformed(e) => write!(f, "Malformed response from Ollama: {e}"),
            Self::Timeout(e) => write!(f, "Request to Ollama timed out: {e}"),
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
        let response = self.http.get(format!("{}/api/tags", self.base_url))
            .timeout(Duration::from_secs(10)).send()
            .map_err(|e| if e.is_timeout() {
                OllamaClientError::Timeout(e.to_string())
            } else { OllamaClientError::Transport(e.to_string()) })?;
        let status = response.status();
        let body = response.text().unwrap_or_default();
        if !status.is_success() {
            return Err(OllamaClientError::Http { status: status.as_u16(), body });
        }
        if let Some(required) = required_model {
            let data: Value = serde_json::from_str(&body)
                .map_err(|e| OllamaClientError::Malformed(e.to_string()))?;
            let found = data.get("models").and_then(Value::as_array).into_iter().flatten()
                .filter_map(|m| m.get("name").and_then(Value::as_str))
                .any(|n| n == required || n.starts_with(&format!("{required}:")));
            if !found {
                return Err(OllamaClientError::Malformed(
                    format!("Model '{required}' was not found in Ollama.")
                ));
            }
        }
        Ok(())
    }    pub fn chat_completion_stream<F>(
        &self, model: &str, messages: &[Value], tools: Option<&[Value]>,
        temperature: f64, mut on_event: F,
    ) -> Result<(Value, Duration), OllamaClientError>
    where F: FnMut(StreamEvent) {
        let mut payload = json!({
            "model": model, "messages": messages,
            "temperature": temperature, "stream": true,
        });
        if let Some(tools) = tools {
            payload["tools"] = Value::Array(tools.to_vec());
        }
        let started = Instant::now();
        let response = self.http
            .post(format!("{}/v1/chat/completions", self.base_url))
            .json(&payload).send()
            .map_err(|e| if e.is_timeout() {
                OllamaClientError::Timeout(format!("after {:?}", self.timeout))
            } else { OllamaClientError::Transport(e.to_string()) })?;

        let status = response.status();
        if !status.is_success() {
            return Err(OllamaClientError::Http {
                status: status.as_u16(), body: response.text().unwrap_or_default()
            });
        }

        let mut content = String::new();
        let mut tool_calls = std::collections::BTreeMap::<usize, Value>::new();
        let mut finish_reason = None;
        let mut usage = None;
        let mut role = "assistant".to_string();

        for line in BufReader::new(response).lines() {
            let line = line.map_err(|e| OllamaClientError::Transport(e.to_string()))?;
            let mut data = line.trim().to_string();
            if data.is_empty() { continue; }
            if let Some(stripped) = data.strip_prefix("data:") {
                data = stripped.trim().to_string();
            }
            if data == "[DONE]" { break; }
            let chunk: Value = match serde_json::from_str(&data) {
                Ok(v) => v, Err(_) => continue
            };
            if let Some(v) = chunk.get("usage") { usage = Some(v.clone()); }
            let Some(choice) = chunk.get("choices").and_then(Value::as_array).and_then(|v| v.first()) else { continue; };
            if let Some(reason) = choice.get("finish_reason").and_then(Value::as_str) {
                finish_reason = Some(reason.to_string());
            }
            let delta = choice.get("delta").cloned().unwrap_or_else(|| json!({}));
            if let Some(value) = delta.get("role").and_then(Value::as_str) { role = value.to_string(); }
            if let Some(piece) = delta.get("content").and_then(Value::as_str) {
                if !piece.is_empty() {
                    content.push_str(piece);
                    on_event(StreamEvent { kind: StreamEventKind::Content(piece.to_string()) });
                }
            }            if let Some(deltas) = delta.get("tool_calls").and_then(Value::as_array) {
                for call in deltas {
                    let index = call.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
                    let entry = tool_calls.entry(index).or_insert_with(|| json!({
                        "id": format!("call_{index}"), "type": "function",
                        "function": {"name": "", "arguments": ""}
                    }));
                    if let Some(id) = call.get("id").and_then(Value::as_str) {
                        entry["id"] = id.into();
                    }
                    let function = call.get("function").cloned().unwrap_or_else(|| json!({}));
                    if let Some(name) = function.get("name").and_then(Value::as_str) {
                        let current = entry["function"]["name"].as_str().unwrap_or_default();
                        entry["function"]["name"] = format!("{current}{name}").into();
                    }
                    if let Some(arguments) = function.get("arguments").and_then(Value::as_str) {
                        let current = entry["function"]["arguments"].as_str().unwrap_or_default();
                        entry["function"]["arguments"] = format!("{current}{arguments}").into();
                    }
                }
                on_event(StreamEvent {
                    kind: StreamEventKind::ToolCallDelta(Value::Array(deltas.clone())),
                });
            }
        }

        let mut message = json!({"role": role, "content": content});
        if !tool_calls.is_empty() {
            message["tool_calls"] = Value::Array(tool_calls.into_values().collect());
        }
        let elapsed = started.elapsed();
        on_event(StreamEvent {
            kind: StreamEventKind::Done {
                message: message.clone(), elapsed, finish_reason, usage
            },
        });
        Ok((message, elapsed))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn normalizes_url() {
        let c = OllamaClient::new("http://localhost:11434///", Duration::from_secs(1)).unwrap();
        assert_eq!(c.base_url, "http://localhost:11434");
    }
}