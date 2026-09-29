use crate::{input::Composer, ui};
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers, MouseEventKind};
use ratatui::{Terminal, backend::CrosstermBackend};
use std::{
    io,
    path::PathBuf,
    time::{Duration, Instant},
};

pub const SLASH_COMMANDS: &[(&str, &str)] = &[
    ("/help", "Show available commands and shortcuts"),
    ("/clear", "Clear the terminal screen"),
    ("/model", "Switch model: /model <name>"),
    (
        "/models",
        "List installed models and pick one (arrows + Enter)",
    ),
    ("/workspace", "Change workspace: /workspace <path>"),
    ("/stats", "Show stats from the last execution"),
    ("/tree", "Show workspace directory tree"),
    ("/ls", "List workspace directory contents"),
    ("/new", "Start a new conversation (clear history)"),
    ("/exit", "Exit Echo (alias: /quit)"),
    ("/quit", "Exit Echo (alias: /exit)"),
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Overlay {
    None,
    Help,
    ModelPicker,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ScrollState {
    pub offset: usize,
    pub max_offset: usize,
    pub viewport_height: usize,
    pub follow_end: bool,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CompletionState {
    pub selected: usize,
    pub visible: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessageRole {
    User,
    Assistant,
}

#[derive(Debug, Clone)]
pub struct Message {
    pub role: MessageRole,
    pub content: String,
}

#[derive(Debug)]
pub struct EchoApp {
    pub running: bool,
    pub composer: Composer,
    pub messages: Vec<Message>,
    pub overlay: Overlay,
    pub completion: CompletionState,
    pub selected_model: usize,
    pub scroll: ScrollState,
    pub model: String,
    pub workspace: PathBuf,
    last_activity: Instant,
}

impl EchoApp {
    pub fn new() -> Self {
        Self {
            running: true,
            composer: Composer::new(),
            messages: Vec::new(),
            overlay: Overlay::None,
            completion: CompletionState::default(),
            selected_model: 0,
            scroll: ScrollState {
                follow_end: true,
                ..ScrollState::default()
            },
            model: "qwen2.5:3b-instruct".to_string(),
            workspace: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
            last_activity: Instant::now(),
        }
    }

    pub fn run(&mut self, terminal: &mut Terminal<CrosstermBackend<io::Stdout>>) -> io::Result<()> {
        while self.running {
            terminal
                .draw(|frame| ui::draw(frame, self))
                .map_err(io::Error::other)?;

            if event::poll(Duration::from_millis(50))? {
                self.handle_event(event::read()?);
            }
        }

        Ok(())
    }

    fn handle_event(&mut self, event: Event) {
        self.last_activity = Instant::now();

        match event {
            Event::Key(key) => self.handle_key(key),
            Event::Mouse(mouse) => match mouse.kind {
                MouseEventKind::ScrollUp => self.scroll_by(-3),
                MouseEventKind::ScrollDown => self.scroll_by(3),
                _ => {}
            },
            _ => {}
        }
    }

    fn handle_key(&mut self, key: KeyEvent) {
        if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
            self.running = false;
            return;
        }

        if self.overlay == Overlay::ModelPicker {
            self.handle_model_picker_key(key);
            return;
        }

        if self.completion_active() {
            match key.code {
                KeyCode::Esc => {
                    self.completion.selected = 0;
                    self.completion.visible = false;
                    return;
                }
                KeyCode::Up => {
                    self.completion.selected = self.completion.selected.saturating_sub(1);
                    return;
                }
                KeyCode::Down => {
                    let count = self.matching_commands().len();
                    if count > 0 {
                        self.completion.selected =
                            (self.completion.selected + 1).min(count.saturating_sub(1));
                    }
                    return;
                }
                KeyCode::Tab => {
                    self.accept_completion();
                    return;
                }
                _ => {}
            }
        }

        match self.overlay {
            Overlay::Help => match key.code {
                KeyCode::Esc | KeyCode::Char('?') => {
                    self.overlay = Overlay::None;
                    self.scroll.follow_end = true;
                }
                KeyCode::PageUp => self.scroll_by(-(self.page_size() as isize)),
                KeyCode::PageDown => self.scroll_by(self.page_size() as isize),
                _ => {}
            },
            Overlay::None => self.handle_composer_key(key),
            Overlay::ModelPicker => unreachable!(),
        }
    }

    fn handle_model_picker_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Esc => {
                self.overlay = Overlay::None;
                self.scroll.follow_end = true;
            }
            KeyCode::Up => {
                self.selected_model = self.selected_model.saturating_sub(1);
            }
            KeyCode::Down => {
                self.selected_model = (self.selected_model + 1).min(1);
            }
            KeyCode::Enter => {
                self.model = if self.selected_model == 0 {
                    "qwen2.5:3b-instruct".into()
                } else {
                    "qwen2.5-coder:3b".into()
                };
                self.overlay = Overlay::None;
                self.scroll.follow_end = true;
            }
            _ => {}
        }
    }

    fn handle_composer_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Char('?') if self.composer.is_empty() => {
                self.overlay = Overlay::Help;
                self.completion.visible = false;
                self.scroll.follow_end = false;
                self.scroll.offset = 0;
            }
            KeyCode::Char('/') if self.composer.is_empty() => {
                self.composer.insert('/');
                self.completion.selected = 0;
                self.completion.visible = true;
                self.scroll.follow_end = true;
            }
            KeyCode::Enter => self.submit(),
            KeyCode::Backspace => {
                self.composer.backspace();
                self.completion.selected = 0;
                self.completion.visible = self.composer.text().starts_with('/');
            }
            KeyCode::Delete => {
                self.composer.delete();
                self.completion.selected = 0;
                self.completion.visible = self.composer.text().starts_with('/');
            }
            KeyCode::Left => self.composer.move_left(),
            KeyCode::Right => self.composer.move_right(),
            KeyCode::Home if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.scroll_to_start()
            }
            KeyCode::End if key.modifiers.contains(KeyModifiers::CONTROL) => self.scroll_to_end(),
            KeyCode::Home => self.composer.home(),
            KeyCode::End => self.composer.end(),
            KeyCode::PageUp => self.scroll_by(-(self.page_size() as isize)),
            KeyCode::PageDown => self.scroll_by(self.page_size() as isize),
            KeyCode::Char(ch) => {
                self.composer.insert(ch);
                self.completion.selected = 0;
                self.completion.visible = self.composer.text().starts_with('/');
            }
            _ => {}
        }
    }

    pub fn matching_commands(&self) -> Vec<(&'static str, &'static str)> {
        let query = self.composer.text().to_lowercase();

        if !query.starts_with('/') || query.contains(' ') {
            return Vec::new();
        }

        SLASH_COMMANDS
            .iter()
            .copied()
            .filter(|(command, _)| command.starts_with(&query))
            .collect()
    }

    pub fn completion_active(&self) -> bool {
        self.completion.visible && !self.matching_commands().is_empty()
    }

    fn accept_completion(&mut self) {
        if let Some((command, _)) = self.matching_commands().get(self.completion.selected) {
            self.composer.clear();
            for ch in command.chars() {
                self.composer.insert(ch);
            }
            self.completion.visible = true;
        }
    }

    fn submit(&mut self) {
        let text = self.composer.text();

        if text.trim().is_empty() {
            return;
        }

        if let Some(command) = text.strip_prefix('/') {
            self.handle_command(command.trim(), text.clone());
        } else {
            self.messages.push(Message {
                role: MessageRole::User,
                content: text,
            });
            self.messages.push(Message {
                role: MessageRole::Assistant,
                content: "Echo base is running. The inference backend is not connected yet.".into(),
            });
        }

        self.composer.clear();
        self.completion.selected = 0;
        self.completion.visible = false;
        self.overlay = Overlay::None;
        self.scroll.follow_end = true;
    }

    fn handle_command(&mut self, command: &str, raw: String) {
        self.messages.push(Message {
            role: MessageRole::User,
            content: raw,
        });

        match command {
            "new" | "clear" => {
                self.messages.clear();
                self.overlay = Overlay::None;
            }
            "quit" | "exit" | "q" => self.running = false,
            "help" => {
                self.overlay = Overlay::Help;
                self.scroll.follow_end = false;
                self.scroll.offset = 0;
            }
            "models" => {
                self.selected_model = 0;
                self.overlay = Overlay::ModelPicker;
                self.completion.visible = false;
                self.scroll.follow_end = true;
            }
            _ => self.messages.push(Message {
                role: MessageRole::Assistant,
                content: format!("Unknown command: /{command}"),
            }),
        }
    }

    pub fn scroll_by(&mut self, delta: isize) {
        if self.scroll.max_offset == 0 {
            self.scroll.offset = 0;
            return;
        }

        self.scroll.follow_end = false;

        if delta.is_negative() {
            self.scroll.offset = self.scroll.offset.saturating_sub(delta.unsigned_abs());
        } else {
            self.scroll.offset = self
                .scroll
                .offset
                .saturating_add(delta as usize)
                .min(self.scroll.max_offset);
        }

        if self.scroll.offset >= self.scroll.max_offset {
            self.scroll.follow_end = true;
        }
    }

    pub fn scroll_to_start(&mut self) {
        self.scroll.follow_end = false;
        self.scroll.offset = 0;
    }

    pub fn scroll_to_end(&mut self) {
        self.scroll.follow_end = true;
        self.scroll.offset = self.scroll.max_offset;
    }

    fn page_size(&self) -> usize {
        self.scroll.viewport_height.max(1)
    }
}
