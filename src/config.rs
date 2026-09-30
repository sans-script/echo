use serde::{Deserialize, Serialize};
use std::{env, fs, io, path::PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SavedConfig {
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    workspace: Option<String>,
}

#[derive(Debug, Clone)]
pub struct EchoConfig {
    pub ollama_url: String,
    pub model: String,
    pub workspace: PathBuf,
    pub max_iterations: usize,
    pub temperature: f64,
    pub request_timeout: f64,
    pub max_tool_output_chars: usize,
    pub system_prompt: String,
}

impl Default for EchoConfig {
    fn default() -> Self {
        Self {
            ollama_url: "http://localhost:11434".to_string(),
            model: "qwen2.5-coder".to_string(),
            workspace: env::current_dir()
                .map(|p| p.join("workspace"))
                .unwrap_or_else(|_| PathBuf::from("./workspace")),
            max_iterations: 10,
            temperature: 0.0,
            request_timeout: 60.0,
            max_tool_output_chars: 8000,
            system_prompt: "You are Echo, a local AI support assistant running inside a terminal. Be practical, concise, technically capable, and honest.".to_string(),
        }
    }
}

impl EchoConfig {
    pub fn load() -> Self {
        let defaults = Self::default();
        let Some(path) = config_path() else {
            return defaults;
        };

        let saved = match fs::read_to_string(path) {
            Ok(text) => match serde_json::from_str::<SavedConfig>(&text) {
                Ok(saved) => saved,
                Err(_) => return defaults,
            },
            Err(_) => return defaults,
        };

        let model = saved
            .model
            .filter(|m| !m.trim().is_empty())
            .unwrap_or(defaults.model);
        let workspace = saved
            .workspace
            .map(PathBuf::from)
            .filter(|p| p.is_dir())
            .unwrap_or(defaults.workspace);

        Self {
            ollama_url: defaults.ollama_url,
            model,
            workspace,
            max_iterations: defaults.max_iterations,
            temperature: defaults.temperature,
            request_timeout: defaults.request_timeout,
            max_tool_output_chars: defaults.max_tool_output_chars,
            system_prompt: defaults.system_prompt,
        }
    }

    pub fn save(&self) -> io::Result<()> {
        let Some(path) = config_path() else {
            return Ok(());
        };

        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }

        let data = SavedConfig {
            model: Some(self.model.clone()),
            workspace: Some(self.workspace.to_string_lossy().into_owned()),
        };

        let text = serde_json::to_string_pretty(&data).map_err(io::Error::other)?;
        fs::write(
            path,
            format!(
                "{text}
"
            ),
        )
    }
}

fn config_path() -> Option<PathBuf> {
    env::var_os("HOME")
        .map(PathBuf::from)
        .or_else(|| env::var_os("USERPROFILE").map(PathBuf::from))
        .map(|home| home.join(".echo").join("config.json"))
}

#[cfg(test)]
mod tests {
    use super::EchoConfig;

    #[test]
    fn default_config_has_expected_model() {
        assert_eq!(EchoConfig::default().model, "qwen2.5-coder");
    }
}
