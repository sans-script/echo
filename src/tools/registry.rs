use serde_json::{Map, Value};
use std::{collections::BTreeMap, time::Instant};

pub type ToolHandler = fn(&Map<String, Value>) -> Result<String, String>;

#[derive(Clone)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: Value,
    pub requires_confirmation: bool,
    pub handler: ToolHandler,
}

impl ToolDefinition {
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: Value,
        requires_confirmation: bool,
        handler: ToolHandler,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
            requires_confirmation,
            handler,
        }
    }

    pub fn openai_definition(&self) -> Value {
        serde_json::json!({
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        })
    }
}

#[derive(Debug, Clone)]
pub struct ToolExecutionRecord {
    pub call_id: String,
    pub tool_name: String,
    pub arguments: Map<String, Value>,
    pub raw_arguments: String,
    pub result: String,
    pub success: bool,
    pub duration_seconds: f64,
    pub error: Option<String>,
}

pub struct ToolRegistry {
    tools: BTreeMap<String, ToolDefinition>,
    pub max_output_chars: usize,
    pub execution_history: Vec<ToolExecutionRecord>,
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new(8000)
    }
}

impl ToolRegistry {
    pub fn new(max_output_chars: usize) -> Self {
        Self {
            tools: BTreeMap::new(),
            max_output_chars,
            execution_history: Vec::new(),
        }
    }

    pub fn register(&mut self, definition: ToolDefinition) -> Result<(), String> {
        if self.tools.contains_key(&definition.name) {
            return Err(format!("Tool '{}' is already registered.", definition.name));
        }
        self.tools.insert(definition.name.clone(), definition);
        Ok(())
    }

    pub fn get(&self, name: &str) -> Option<&ToolDefinition> {
        self.tools.get(name)
    }

    pub fn requires_confirmation(&self, name: &str) -> bool {
        self.get(name)
            .map(|tool| tool.requires_confirmation)
            .unwrap_or(false)
    }

    pub fn definitions(&self) -> Vec<Value> {
        self.tools
            .values()
            .map(ToolDefinition::openai_definition)
            .collect()
    }

    pub fn validate_and_parse_args(
        &self,
        tool_name: &str,
        raw_arguments: &str,
    ) -> Result<Map<String, Value>, String> {
        let tool = self.get(tool_name).ok_or_else(|| {
            let available = self.tools.keys().cloned().collect::<Vec<_>>().join(", ");
            format!("Unknown tool '{tool_name}'. Available tools: [{available}]")
        })?;

        let args: Value = if raw_arguments.trim().is_empty() {
            Value::Object(Map::new())
        } else {
            serde_json::from_str(raw_arguments)
                .map_err(|e| format!("Invalid JSON in tool arguments: {e}"))?
        };

        let mut args = args
            .as_object()
            .cloned()
            .ok_or_else(|| "Arguments must be a JSON object.".to_string())?;

        let schema = tool
            .parameters
            .as_object()
            .ok_or_else(|| "Tool parameter schema must be a JSON object.".to_string())?;

        let properties = schema
            .get("properties")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();

        let required = schema
            .get("required")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();

        let missing = required
            .iter()
            .filter_map(Value::as_str)
            .filter(|name| !args.contains_key(*name))
            .collect::<Vec<_>>();

        if !missing.is_empty() {
            return Err(format!(
                "Missing required parameter(s): {}",
                missing.join(", ")
            ));
        }

        let unknown = args
            .keys()
            .filter(|name| !properties.contains_key(*name))
            .cloned()
            .collect::<Vec<_>>();

        if !unknown.is_empty() {
            let allowed = properties.keys().cloned().collect::<Vec<_>>().join(", ");
            return Err(format!(
                "Unknown parameter(s): {}. Allowed parameters: {}",
                unknown.join(", "),
                if allowed.is_empty() {
                    "(none)"
                } else {
                    &allowed
                }
            ));
        }

        for (name, value) in args.iter_mut() {
            let Some(expected) = properties
                .get(name)
                .and_then(|schema| schema.get("type"))
                .and_then(Value::as_str)
            else {
                continue;
            };

            *value = coerce_value(value, expected)
                .map_err(|e| format!("Invalid value for parameter '{name}': {e}"))?;
        }

        Ok(args)
    }

    pub fn execute(
        &mut self,
        call_id: impl Into<String>,
        tool_name: &str,
        raw_arguments: &str,
    ) -> ToolExecutionRecord {
        let start = Instant::now();
        let call_id = call_id.into();

        let parsed = self.validate_and_parse_args(tool_name, raw_arguments);
        let (arguments, result, success, error) = match parsed {
            Ok(arguments) => {
                let handler = self.get(tool_name).map(|tool| tool.handler);
                match handler {
                    Some(handler) => match handler(&arguments) {
                        Ok(mut result) => {
                            if result.len() > self.max_output_chars {
                                let total = result.len();
                                result.truncate(self.max_output_chars);
                                result.push_str(&format!(
                                    "
... [Output truncated; total length: {total} characters]"
                                ));
                            }
                            (arguments, result, true, None)
                        }
                        Err(error) => (
                            arguments,
                            format!("Error executing tool '{tool_name}': {error}"),
                            false,
                            Some(error),
                        ),
                    },
                    None => (
                        Map::new(),
                        format!("Unknown tool '{tool_name}'."),
                        false,
                        Some(format!("Unknown tool '{tool_name}'.")),
                    ),
                }
            }
            Err(error) => (Map::new(), format!("Error: {error}"), false, Some(error)),
        };

        let record = ToolExecutionRecord {
            call_id,
            tool_name: tool_name.to_string(),
            arguments,
            raw_arguments: raw_arguments.to_string(),
            result,
            success,
            duration_seconds: start.elapsed().as_secs_f64(),
            error,
        };
        self.execution_history.push(record.clone());
        record
    }
}

fn coerce_value(value: &Value, expected: &str) -> Result<Value, String> {
    match expected {
        "string" => match value {
            Value::String(_) => Ok(value.clone()),
            _ => Err(format!(
                "expected type 'string', got '{}'",
                value_type(value)
            )),
        },
        "integer" => match value {
            Value::Number(n) if n.as_i64().is_some() => Ok(value.clone()),
            Value::String(s) => s
                .trim()
                .parse::<i64>()
                .map(|n| Value::from(n))
                .map_err(|_| format!("could not convert string {s:?} to integer")),
            _ => Err(format!(
                "expected type 'integer', got '{}'",
                value_type(value)
            )),
        },
        "number" => match value {
            Value::Number(_) => Ok(value.clone()),
            Value::String(s) => serde_json::from_str::<Value>(s.trim())
                .ok()
                .filter(|v| v.is_number())
                .ok_or_else(|| format!("could not convert string {s:?} to number")),
            _ => Err(format!(
                "expected type 'number', got '{}'",
                value_type(value)
            )),
        },
        "boolean" => match value {
            Value::Bool(_) => Ok(value.clone()),
            Value::String(s) => match s.trim().to_ascii_lowercase().as_str() {
                "true" => Ok(Value::Bool(true)),
                "false" => Ok(Value::Bool(false)),
                _ => Err(format!(
                    "could not convert string {s:?} to boolean; expected 'true' or 'false'"
                )),
            },
            _ => Err(format!(
                "expected type 'boolean', got '{}'",
                value_type(value)
            )),
        },
        "array" => match value {
            Value::Array(_) => Ok(value.clone()),
            Value::String(s) => serde_json::from_str::<Value>(s.trim())
                .ok()
                .filter(Value::is_array)
                .ok_or_else(|| format!("could not parse string {s:?} as JSON array")),
            _ => Err(format!(
                "expected type 'array', got '{}'",
                value_type(value)
            )),
        },
        "object" => match value {
            Value::Object(_) => Ok(value.clone()),
            Value::String(s) => serde_json::from_str::<Value>(s.trim())
                .ok()
                .filter(Value::is_object)
                .ok_or_else(|| format!("could not parse string {s:?} as JSON object")),
            _ => Err(format!(
                "expected type 'object', got '{}'",
                value_type(value)
            )),
        },
        _ => Ok(value.clone()),
    }
}

fn value_type(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn echo(args: &Map<String, Value>) -> Result<String, String> {
        Ok(args.get("value").unwrap().to_string())
    }

    fn failing(_: &Map<String, Value>) -> Result<String, String> {
        Err("boom".into())
    }

    fn definition(confirm: bool, handler: ToolHandler) -> ToolDefinition {
        ToolDefinition::new(
            "echo",
            "Echo a value.",
            json!({
                "type": "object",
                "properties": {
                    "value": {"type": "string"},
                    "count": {"type": "integer"},
                    "enabled": {"type": "boolean"}
                },
                "required": ["value"],
                "additionalProperties": false
            }),
            confirm,
            handler,
        )
    }

    #[test]
    fn registers_and_rejects_duplicate_names() {
        let mut registry = ToolRegistry::default();
        registry.register(definition(false, echo)).unwrap();
        assert!(registry.register(definition(false, echo)).is_err());
    }

    #[test]
    fn exports_openai_compatible_definition() {
        let mut registry = ToolRegistry::default();
        registry.register(definition(true, echo)).unwrap();
        let definitions = registry.definitions();
        assert_eq!(definitions[0]["type"], "function");
        assert_eq!(definitions[0]["function"]["name"], "echo");
        assert_eq!(
            definitions[0]["function"]["parameters"]["required"][0],
            "value"
        );
        assert!(registry.requires_confirmation("echo"));
    }

    #[test]
    fn validates_required_and_unknown_arguments() {
        let mut registry = ToolRegistry::default();
        registry.register(definition(false, echo)).unwrap();

        assert!(registry.validate_and_parse_args("echo", "{}").is_err());
        assert!(
            registry
                .validate_and_parse_args("echo", r#"{"value":"ok","extra":1}"#)
                .is_err()
        );
    }

    #[test]
    fn coerces_model_generated_scalar_strings() {
        let mut registry = ToolRegistry::default();
        registry.register(definition(false, echo)).unwrap();

        let args = registry
            .validate_and_parse_args("echo", r#"{"value":"ok","count":"3","enabled":"true"}"#)
            .unwrap();

        assert_eq!(args["count"], json!(3));
        assert_eq!(args["enabled"], json!(true));
    }

    #[test]
    fn records_success_and_truncates_output() {
        let mut registry = ToolRegistry::new(4);
        registry.register(definition(false, echo)).unwrap();

        let record = registry.execute("call-1", "echo", r#"{"value":"abcdef"}"#);

        assert!(record.success);
        assert!(record.result.starts_with("\"abc"));
        assert_eq!(registry.execution_history.len(), 1);
    }

    #[test]
    fn records_handler_errors() {
        let mut registry = ToolRegistry::default();
        registry.register(definition(false, failing)).unwrap();

        let record = registry.execute("call-2", "echo", r#"{"value":"ok"}"#);

        assert!(!record.success);
        assert_eq!(record.error.as_deref(), Some("boom"));
        assert!(record.result.contains("Error executing tool"));
    }
}
