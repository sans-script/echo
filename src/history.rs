#[derive(Debug, Default)]
pub struct History {
    entries: Vec<String>,
    position: Option<usize>,
    draft: String,
}

impl History {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, text: &str) {
        let text = text.trim_end_matches('\n');
        if text.trim().is_empty() {
            return;
        }

        if self.entries.last().map(String::as_str) != Some(text) {
            self.entries.push(text.to_string());
        }

        self.position = None;
        self.draft.clear();
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

    #[cfg(test)]
    fn entries(&self) -> &[String] {
        &self.entries
    }
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
    fn consecutive_duplicates_are_ignored() {
        let mut history = History::new();
        history.push("same");
        history.push("same");
        assert_eq!(history.entries(), &["same".to_string()]);
    }
}
