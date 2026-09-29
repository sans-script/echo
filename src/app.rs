use std::{
    io,
    path::PathBuf,
    time::{Duration, Instant},
};

use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers, MouseEventKind};
use ratatui::{backend::Backend, Terminal};

use crate::{input::Composer, ui};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Overlay {
    None,
    Help,
    Commands,
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
    pub scroll: usize,
    pub model: String,
    pub workspace: PathBuf,
    last_activity: Instant,
}

impl EchoApp {
    pub fn new() -> Self {
        let workspace = std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."));

        Self {
            running: true,
            composer: Composer::new(),
            messages: Vec::new(),
            overlay: Overlay::None,
            scroll: 0,
            model: "qwen2.5:3b-instruct".to_string(),
            workspace,
            last_activity: Instant::now(),
        }
    }

    pub fn run<B: Backend>(&mut self, terminal: &mut Terminal<B>) -> io::Result<()> {
        while self.running {
            terminal.draw(|frame| ui::draw(frame, self))?;

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
            Event::Mouse(mouse) => {
                if matches!(mouse.kind, MouseEventKind::ScrollUp) {
                    self.scroll = self.scroll.saturating_add(1);
                } else if matches!(mouse.kind, MouseEventKind::ScrollDown) {
                    self.scroll = self.scroll.saturating_sub(1);
                }
            }
            _ => {}
        }
    }

    fn handle_key(&mut self, key: KeyEvent) {
        if key.modifiers.contains(KeyModifiers::CONTROL)
            && matches!(key.code, KeyCode::Char('c'))
        {
            self.running = false;
            return;
        }

        match self.overlay {
            Overlay::Help | Overlay::Commands => self.handle_overlay_key(key),
            Overlay::None => self.handle_composer_key(key),
        }
    }

    fn handle_overlay_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Esc => self.overlay = Overlay::None,
            KeyCode::Char('?') if self.overlay == Overlay::Help => {
                self.overlay = Overlay::None;
            }
            KeyCode::Char('/') if self.overlay == Overlay::Commands => {
                self.overlay = Overlay::None;
            }
            _ => {}
        }
    }

    fn handle_composer_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Char('?') if self.composer.is_empty() => {
                self.overlay = Overlay::Help;
            }
            KeyCode::Char('/') if self.composer.is_empty() => {
                self.overlay = Overlay::Commands;
            }
            KeyCode::Enter => self.submit(),
            KeyCode::Backspace => self.composer.backspace(),
            KeyCode::Delete => self.composer.delete(),
            KeyCode::Left => self.composer.move_left(),
            KeyCode::Right => self.composer.move_right(),
            KeyCode::Home => self.composer.home(),
            KeyCode::End => self.composer.end(),
            KeyCode::Char(ch) => self.composer.insert(ch),
            _ => {}
        }
    }

    fn submit(&mut self) {
        let text = self.composer.text();

        if text.trim().is_empty() {
            return;
        }

        if let Some(command) = text.strip_prefix('/') {
            self.handle_command(command.trim());
        } else {
            self.messages.push(Message {
                role: MessageRole::User,
                content: text,
            });

            self.messages.push(Message {
                role: MessageRole::Assistant,
                content: "Rust Echo base is running. The inference backend is not connected yet."
                    .to_string(),
            });
        }

        self.composer.clear();
        self.scroll = 0;
    }

    fn handle_command(&mut self, command: &str) {
        match command {
            "new" | "clear" => self.messages.clear(),
            "quit" | "exit" | "q" => self.running = false,
            "help" => self.overlay = Overlay::Help,
            "models" => self.overlay = Overlay::Commands,
            _ => {
                self.messages.push(Message {
                    role: MessageRole::Assistant,
                    content: format!("Unknown command: /{command}"),
                });
            }
        }
    }
}
