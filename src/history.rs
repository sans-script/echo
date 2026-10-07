use serde::Deserialize;
use serde_json::json;
use std::{
    env,
    fs::{self, OpenOptions},
    io::{self, BufRead, Write},
    path::PathBuf,
    time::SystemTime,
};

#[derive(Debug, Default)]
pub struct History {
    entries: Vec<String>,
    position: Option<usize>,
    draft: String,
    file: Option<PathBuf>,
    workspace: Option<String>,
}

#[derive(Debug, Deserialize)]
struct HistoryEntry {
    text: String,
}

impl History {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn load_persistent(workspace: &std::path::Path) -> Self {
        let file = history_path();
        let workspace = workspace.to_string_lossy().into_owned();
        let mut history = Self {
            entries: Vec::new(),
            position: None,
            draft: String::new(),
            file: Some(file.clone()),
            workspace: Some(workspace),
        };

        let Ok(file) = fs::File::open(file) else {
            return history;
        };

        for line in io::BufReader::new(file).lines().map_while(Result::ok) {
            if let Ok(entry) = serde_json::from_str::<HistoryEntry>(&line) {
                if !entry.text.is_empty() {
                    history.entries.push(entry.text);
                }
            }
        }

        history
    }

    pub fn set_workspace(&mut self, workspace: &std::path::Path) {
        self.workspace = Some(workspace.to_string_lossy().into_owned());
    }

    pub fn push(&mut self, text: &str) {
        let raw = text.trim_end_matches('\n');
        let trimmed = raw.trim();

        if trimmed.is_empty()
            || raw.starts_with(' ')
            || matches!(
                trimmed.to_ascii_lowercase().as_str(),
                "/exit" | "/quit" | "exit" | "quit"
            )
        {
            self.clear_navigation();
            return;
        }

        if self.entries.last().map(String::as_str) == Some(trimmed) {
            self.clear_navigation();
            return;
        }

        self.entries.push(trimmed.to_string());

        if let Some(file) = &self.file {
            if let Some(parent) = file.parent() {
                let _ = fs::create_dir_all(parent);
            }

            let entry = json!({
                "text": trimmed,
                "timestamp": timestamp(),
                "workspace": self.workspace.as_deref().unwrap_or(""),
            });

            if let Ok(line) = serde_json::to_string(&entry) {
                if let Ok(mut out) = OpenOptions::new().create(true).append(true).open(file) {
                    let _ = writeln!(out, "{line}");
                }
            }
        }

        self.clear_navigation();
    }

    pub fn previous(&mut self, current: &str) -> Option<String> {
        if self.entries.is_empty() {
            return None;
        }

        if self.position.is_none() {
            self.draft = current.to_string();
            self.position = Some(self.entries.len() - 1);
        } else if self.position.unwrap() > 0 {
            self.position = Some(self.position.unwrap() - 1);
        }

        self.position.map(|index| self.entries[index].clone())
    }

    pub fn next(&mut self) -> Option<String> {
        let position = self.position?;

        if position + 1 < self.entries.len() {
            self.position = Some(position + 1);
            return Some(self.entries[position + 1].clone());
        }

        self.position = None;
        Some(self.draft.clone())
    }

    pub fn clear_navigation(&mut self) {
        self.position = None;
        self.draft.clear();
    }

    /// Newest entry older than index `before` that contains `query`
    /// (case-insensitive), for Ctrl+R reverse search.
    pub fn search(&self, query: &str, before: usize) -> Option<(usize, &str)> {
        let query = query.to_lowercase();
        self.entries[..before.min(self.entries.len())]
            .iter()
            .enumerate()
            .rev()
            .find(|(_, entry)| entry.to_lowercase().contains(&query))
            .map(|(index, entry)| (index, entry.as_str()))
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[cfg(test)]
    fn entries(&self) -> &[String] {
        &self.entries
    }
}

fn history_path() -> PathBuf {
    env::var_os("HOME")
        .map(PathBuf::from)
        .or_else(|| env::var_os("USERPROFILE").map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".echo")
        .join("history.jsonl")
}

fn timestamp() -> String {
    chrono::DateTime::<chrono::Utc>::from(SystemTime::now()).to_rfc3339()
}

#[cfg(test)]
mod tests {
    use super::History;

    #[test]
    fn history_navigates_previous_and_next() {
        let mut history = History::new();
        history.push("first");
        history.push("second");

        assert_eq!(history.previous("draft").as_deref(), Some("second"));
        assert_eq!(history.previous("second").as_deref(), Some("first"));
        assert_eq!(history.next().as_deref(), Some("second"));
        assert_eq!(history.next().as_deref(), Some("draft"));
    }

    #[test]
    fn reverse_search_walks_older_matches() {
        let mut history = History::new();
        history.push("list files");
        history.push("read a.txt");
        history.push("List the workspace");

        let (newest, text) = history.search("list", history.len()).unwrap();
        assert_eq!(text, "List the workspace");
        let (_, older) = history.search("list", newest).unwrap();
        assert_eq!(older, "list files");
        assert!(history.search("missing", history.len()).is_none());
    }

    #[test]
    fn consecutive_duplicates_are_ignored() {
        let mut history = History::new();
        history.push("same");
        history.push("same");
        assert_eq!(history.entries(), &["same".to_string()]);
    }
}
