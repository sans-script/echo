use crate::{
    clipboard,
    config::EchoConfig,
    history::History,
    input::Composer,
    logo::LogoAnimator,
    ollama::{InstalledModel, normalize_model_name},
    orchestrator::{
        EchoOrchestrator, OrchestratorEvent, OrchestratorResult, RunStopReason, format_stats,
    },
    tools::{
        filesystem::{FilesystemSandbox, line_of, strip_line_numbers},
        registry::{ToolExecutionRecord, ToolRegistry},
    },
    ui,
    workspace_helpers::display_path,
};
use crossterm::event::{
    self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers, MouseButton, MouseEvent,
    MouseEventKind,
};
use ratatui::{Terminal, backend::CrosstermBackend};
use serde_json::Value;
use std::{
    io,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver, Sender, TryRecvError},
    },
    thread,
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
    ("/tree", "Show the workspace tree: /tree [path] [--depth=N]"),
    ("/ls", "List one level as a tree: /ls [path]"),
    ("/new", "Start a new conversation (clear history)"),
    ("/exit", "Exit Echo (alias: /quit)"),
    ("/quit", "Exit Echo (alias: /exit)"),
];

/// Frame time of the UI loop; matches the 30 fps logo animation.
const FRAME_TIME: Duration = Duration::from_millis(33);
/// Pastes larger than this are shown as a placeholder in the composer.
const PASTE_PLACEHOLDER_LINES: usize = 5;
const PASTE_PLACEHOLDER_CHARS: usize = 300;
/// Tool results longer than this are cut in the transcript (except `tree`).
const TOOL_RESULT_LINES: usize = 15;
const MODEL_PICKER_ROWS: usize = 12;
const BUSY_MESSAGE: &str = "Echo is still working. Press Esc to cancel first.";
pub const CONFIRM_QUESTION: &str = "Do you want to continue? [Y/n] ";

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

/// How a transcript entry is rendered. Mirrors the colors of the Python UI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessageRole {
    /// The user's input, echoed with a "> " prefix.
    User,
    /// Model output, with a "> " prefix.
    Assistant,
    /// Tool calls and results (gray).
    Tool,
    /// Gray first line, white body (the `tree` tool result).
    ToolResult,
    /// Plain text in the default color (/ls, /tree, confirmations).
    Plain,
    /// Gray text (stats, hints).
    Muted,
    /// Green text.
    Success,
    /// Yellow first line, gray body (usage messages).
    Usage,
    /// Red first line, gray body.
    Error,
    /// The command and shortcut reference.
    Help,
    /// An answered confirmation request, colored like the live prompt.
    Confirmation,
}

#[derive(Debug, Clone)]
pub struct Message {
    pub role: MessageRole,
    pub content: String,
}

pub struct ModelPicker {
    pub models: Vec<InstalledModel>,
    pub selected: usize,
}

/// A tool waiting for the user's approval before it runs.
pub struct Confirmation {
    /// apt-style summary of what will change.
    pub details: String,
    /// What the user typed so far: `Some(true)` for "y", `Some(false)` for "n".
    pub choice: Option<bool>,
    reply: Sender<bool>,
}

/// Text selected with the mouse, in screen cells (column, row).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Selection {
    pub anchor: (u16, u16),
    pub head: (u16, u16),
    pub dragging: bool,
    /// Set when the drag ends; the next frame copies the selected text.
    pub copy: bool,
}

/// How long footer notices such as "Copied N characters" stay visible.
pub const NOTICE_TIME: Duration = Duration::from_secs(2);

/// Ctrl+R reverse history search.
pub struct HistorySearch {
    pub query: String,
    /// Index of the current match in the history, if any.
    pub found: Option<usize>,
    pub text: String,
}

enum WorkerMessage {
    Event(OrchestratorEvent),
    Confirm {
        tool_name: String,
        arguments: Value,
        reply: Sender<bool>,
    },
    Finished(Box<EchoOrchestrator>, OrchestratorResult),
}

struct PendingRun {
    events: Receiver<WorkerMessage>,
    cancel: Arc<AtomicBool>,
    /// Index in `messages` of the assistant reply currently being streamed.
    reply: Option<usize>,
}

pub struct EchoApp {
    pub running: bool,
    pub composer: Composer,
    pub history: History,
    pub messages: Vec<Message>,
    pub overlay: Overlay,
    pub completion: CompletionState,
    pub model_picker: ModelPicker,
    pub scroll: ScrollState,
    pub model: String,
    pub ollama_url: String,
    pub workspace: PathBuf,
    /// `None` while a run owns the orchestrator on the worker thread.
    pub orchestrator: Option<EchoOrchestrator>,
    pub logo: LogoAnimator,
    pub confirmation: Option<Confirmation>,
    pub search: Option<HistorySearch>,
    /// Set while waiting for the model; drives the "Loading..." spinner.
    pub waiting_since: Option<Instant>,
    /// Show tool calls, tool results and stats in the transcript.
    pub verbose: bool,
    pub selection: Option<Selection>,
    /// Short message shown in the footer for `NOTICE_TIME`.
    pub notice: Option<(String, Instant)>,
    last_stats: Option<String>,
    pastes: Vec<(String, String)>,
    run: Option<PendingRun>,
    warm_up: Option<Receiver<()>>,
}

impl EchoApp {
    #[cfg(test)]
    pub fn new() -> Self {
        Self::with_config(EchoConfig::load(), true)
    }

    pub fn with_config(config: EchoConfig, verbose: bool) -> Self {
        let workspace = config.workspace.clone();
        let history = History::load_persistent(&workspace);
        let mut tools = ToolRegistry::new(config.max_tool_output_chars);
        let _ = crate::tools::register_all(&mut tools, &workspace);
        let mut orchestrator = EchoOrchestrator::new(config.clone(), tools)
            .expect("failed to initialize Ollama client");
        orchestrator.tool_output_visible = verbose;

        Self {
            running: true,
            composer: Composer::new(),
            history,
            messages: Vec::new(),
            overlay: Overlay::None,
            completion: CompletionState::default(),
            model_picker: ModelPicker {
                models: Vec::new(),
                selected: 0,
            },
            scroll: ScrollState {
                follow_end: true,
                ..ScrollState::default()
            },
            model: config.model,
            ollama_url: config.ollama_url,
            workspace,
            orchestrator: Some(orchestrator),
            logo: LogoAnimator::new(verbose),
            confirmation: None,
            search: None,
            waiting_since: None,
            verbose,
            selection: None,
            notice: None,
            last_stats: None,
            pastes: Vec::new(),
            run: None,
            warm_up: None,
        }
    }

    pub fn run(&mut self, terminal: &mut Terminal<CrosstermBackend<io::Stdout>>) -> io::Result<()> {
        self.start_warm_up();

        while self.running {
            self.poll_background();
            self.logo.tick(Instant::now());

            // The TUI redraws frequently enough to drive the software cursor
            // without exposing or depending on the terminal's native cursor.
            self.composer.tick_cursor();

            let mut copied = None;
            terminal
                .draw(|frame| {
                    ui::draw(frame, self);
                    if let Some(selection) = &mut self.selection {
                        ui::highlight_selection(frame.buffer_mut(), selection);
                        if std::mem::take(&mut selection.copy) {
                            copied = Some(ui::selected_text(frame.buffer_mut(), selection));
                        }
                    }
                })
                .map_err(io::Error::other)?;
            if let Some(text) = copied.filter(|text| !text.trim().is_empty()) {
                clipboard::copy(terminal.backend_mut(), &text);
                self.notice = Some((
                    format!("Copied {} characters", text.chars().count()),
                    Instant::now(),
                ));
            }

            if event::poll(FRAME_TIME)? {
                // Drain everything already queued: on Windows a paste arrives
                // as a burst of key events rather than a single paste event.
                let mut batch = vec![event::read()?];
                while event::poll(Duration::ZERO)? {
                    batch.push(event::read()?);
                }
                self.handle_events(batch);
            }
        }

        Ok(())
    }

    fn handle_events(&mut self, batch: Vec<Event>) {
        if let Some(text) = paste_burst(&batch) {
            self.handle_paste(text);
            return;
        }
        for event in batch {
            self.handle_event(event);
        }
    }

    fn handle_event(&mut self, event: Event) {
        match event {
            Event::Key(key) => self.handle_key(key),
            Event::Paste(text) => self.handle_paste(text),
            Event::Mouse(mouse) => self.handle_mouse(mouse),
            _ => {}
        }
    }

    /// The wheel scrolls the conversation. Since capturing the mouse turns
    /// off the terminal's own selection, Echo selects text itself: drag with
    /// the left button, and the text is copied when the button is released.
    fn handle_mouse(&mut self, mouse: MouseEvent) {
        let at = (mouse.column, mouse.row);
        match mouse.kind {
            MouseEventKind::ScrollUp => {
                self.selection = None;
                self.scroll_by(-3);
            }
            MouseEventKind::ScrollDown => {
                self.selection = None;
                self.scroll_by(3);
            }
            MouseEventKind::Down(MouseButton::Left) => {
                self.selection = Some(Selection {
                    anchor: at,
                    head: at,
                    dragging: true,
                    copy: false,
                });
            }
            MouseEventKind::Drag(MouseButton::Left) => {
                if let Some(selection) = self.selection.as_mut().filter(|s| s.dragging) {
                    selection.head = at;
                }
            }
            MouseEventKind::Up(MouseButton::Left) => match self.selection.as_mut() {
                // A plain click clears the selection.
                Some(selection) if selection.anchor == at && selection.head == at => {
                    self.selection = None;
                }
                Some(selection) => {
                    selection.head = at;
                    selection.dragging = false;
                    selection.copy = true;
                }
                None => {}
            },
            _ => {}
        }
    }

    fn handle_key(&mut self, key: KeyEvent) {
        // Windows reports both the press and the release of every key.
        if key.kind == KeyEventKind::Release {
            return;
        }
        self.selection = None;
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);

        if self.confirmation.is_some() {
            self.handle_confirmation_key(key);
            return;
        }

        if self.search.is_some() {
            self.handle_search_key(key);
            return;
        }

        if self.overlay == Overlay::ModelPicker {
            self.handle_model_picker_key(key);
            return;
        }

        if ctrl && key.code == KeyCode::Char('c') {
            if !self.composer.is_empty() {
                self.clear_composer();
            } else if !self.cancel_run() {
                self.running = false;
            }
            return;
        }

        if ctrl && key.code == KeyCode::Char('d') {
            if self.composer.is_empty() {
                self.running = false;
            } else {
                self.composer.delete();
            }
            return;
        }

        if key.code == KeyCode::Esc
            && self.overlay == Overlay::None
            && !self.completion_active()
            && self.cancel_run()
        {
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
                // Enter runs the highlighted command, not the partial text.
                KeyCode::Enter if key.modifiers.is_empty() => {
                    if let Some((command, _)) =
                        self.matching_commands().get(self.completion.selected).copied()
                    {
                        if matches!(command, "/model" | "/workspace") {
                            // These need an argument: let the user type it.
                            self.replace_composer(&format!("{command} "));
                            self.completion.visible = false;
                        } else {
                            self.replace_composer(command);
                            self.submit();
                        }
                        return;
                    }
                }
                _ => {}
            }
        }

        if self.overlay == Overlay::Help {
            match key.code {
                KeyCode::Esc => {
                    self.overlay = Overlay::None;
                    self.scroll.follow_end = true;
                }
                KeyCode::Char('?') if self.composer.is_empty() => {
                    self.overlay = Overlay::None;
                    self.scroll.follow_end = true;
                }
                KeyCode::Char('/') if self.composer.is_empty() => {
                    // Typing '/' while help is visible switches directly to
                    // command completion, just like Python's completion state
                    // disables the shortcuts panel.
                    self.overlay = Overlay::None;
                    self.handle_composer_key(key);
                }
                KeyCode::PageUp => self.scroll_by(-(self.page_size() as isize)),
                KeyCode::PageDown => self.scroll_by(self.page_size() as isize),
                KeyCode::Home if ctrl => self.scroll_to_start(),
                KeyCode::End if ctrl => self.scroll_to_end(),
                _ => self.handle_composer_key(key),
            }
            return;
        }

        self.handle_composer_key(key);
    }

    fn handle_confirmation_key(&mut self, key: KeyEvent) {
        let Some(confirmation) = self.confirmation.as_mut() else {
            return;
        };
        match key.code {
            KeyCode::Char('y' | 'Y') => confirmation.choice = Some(true),
            KeyCode::Char('n' | 'N') => confirmation.choice = Some(false),
            KeyCode::Backspace | KeyCode::Delete => confirmation.choice = None,
            // An empty answer means yes, as in apt's [Y/n].
            KeyCode::Enter => {
                let answer = confirmation.choice.unwrap_or(true);
                self.resolve_confirmation(answer);
            }
            KeyCode::Esc => self.resolve_confirmation(false),
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.resolve_confirmation(false)
            }
            _ => {}
        }
    }

    fn resolve_confirmation(&mut self, answer: bool) {
        let Some(confirmation) = self.confirmation.take() else {
            return;
        };
        let _ = confirmation.reply.send(answer);
        self.messages.push(Message {
            role: MessageRole::Confirmation,
            content: format!(
                "{}\n{CONFIRM_QUESTION}{}",
                confirmation.details,
                if answer { "y" } else { "n" }
            ),
        });
        self.composer.force_cursor_visible();
        self.scroll.follow_end = true;
    }

    fn handle_search_key(&mut self, key: KeyEvent) {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        let Some(search) = self.search.as_mut() else {
            return;
        };
        match key.code {
            KeyCode::Char('r') if ctrl => {
                // Next older match for the same query.
                let before = search.found.unwrap_or(self.history.len());
                if let Some((index, text)) = self.history.search(&search.query, before) {
                    search.found = Some(index);
                    search.text = text.to_string();
                }
            }
            KeyCode::Char('c' | 'g') if ctrl => self.search = None,
            KeyCode::Esc => self.search = None,
            KeyCode::Enter | KeyCode::Left | KeyCode::Right | KeyCode::Home | KeyCode::End => {
                let text = std::mem::take(&mut search.text);
                self.search = None;
                if !text.is_empty() {
                    self.replace_composer(&text);
                }
            }
            KeyCode::Backspace => {
                search.query.pop();
                self.refresh_search();
            }
            KeyCode::Char(ch) if !ctrl => {
                search.query.push(ch);
                self.refresh_search();
            }
            _ => {}
        }
    }

    fn refresh_search(&mut self) {
        let Some(search) = self.search.as_mut() else {
            return;
        };
        match self.history.search(&search.query, self.history.len()) {
            Some((index, text)) => {
                search.found = Some(index);
                search.text = text.to_string();
            }
            None => {
                search.found = None;
                search.text.clear();
            }
        }
    }

    fn handle_model_picker_key(&mut self, key: KeyEvent) {
        let count = self.model_picker.models.len().max(1);
        match key.code {
            KeyCode::Esc | KeyCode::Char('q') => self.close_model_picker(),
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.close_model_picker()
            }
            KeyCode::Up | KeyCode::Char('k') => {
                self.model_picker.selected = (self.model_picker.selected + count - 1) % count;
            }
            KeyCode::Down | KeyCode::Char('j') => {
                self.model_picker.selected = (self.model_picker.selected + 1) % count;
            }
            KeyCode::Enter => {
                let chosen = self
                    .model_picker
                    .models
                    .get(self.model_picker.selected)
                    .map(|model| model.name.clone());
                self.close_model_picker();
                if let Some(model) = chosen {
                    if normalize_model_name(&model) != normalize_model_name(&self.model) {
                        if let Err(message) = self.set_model(&model) {
                            self.push(MessageRole::Muted, message);
                        }
                    }
                }
            }
            _ => {}
        }
    }

    fn close_model_picker(&mut self) {
        self.overlay = Overlay::None;
        self.model_picker.models.clear();
        self.scroll.follow_end = true;
    }

    fn handle_composer_key(&mut self, key: KeyEvent) {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        let alt = key.modifiers.contains(KeyModifiers::ALT);
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
            // Ctrl+J inserts a line break instead of submitting.
            KeyCode::Char('j') if ctrl && !alt => {
                self.history.clear_navigation();
                self.composer.insert('\n');
                self.completion.visible = false;
            }
            KeyCode::Char('r') if ctrl && !alt => {
                self.search = Some(HistorySearch {
                    query: String::new(),
                    found: None,
                    text: String::new(),
                });
            }
            KeyCode::Enter if key.modifiers.intersects(KeyModifiers::SHIFT | KeyModifiers::ALT) => {
                self.history.clear_navigation();
                self.composer.insert('\n');
            }
            KeyCode::Enter => self.submit(),
            KeyCode::Backspace => {
                self.history.clear_navigation();
                self.composer.backspace();
                self.completion.selected = 0;
                self.completion.visible = self.composer.text().starts_with('/');
            }
            KeyCode::Delete => {
                self.history.clear_navigation();
                self.composer.delete();
                self.completion.selected = 0;
                self.completion.visible = self.composer.text().starts_with('/');
            }
            KeyCode::Left => {
                self.history.clear_navigation();
                self.composer.move_left();
            }
            KeyCode::Right => {
                self.history.clear_navigation();
                self.composer.move_right();
            }
            KeyCode::Home if ctrl => self.scroll_to_start(),
            KeyCode::End if ctrl => self.scroll_to_end(),
            KeyCode::Home => {
                self.history.clear_navigation();
                self.composer.home();
            }
            KeyCode::End => {
                self.history.clear_navigation();
                self.composer.end();
            }
            KeyCode::PageUp => self.scroll_by(-(self.page_size() as isize)),
            KeyCode::PageDown => self.scroll_by(self.page_size() as isize),
            KeyCode::Up => {
                if let Some(text) = self.history.previous(&self.composer.text()) {
                    self.replace_composer(&text);
                    self.completion.visible = text.starts_with('/');
                    self.completion.selected = 0;
                }
            }
            KeyCode::Down => {
                if let Some(text) = self.history.next() {
                    self.replace_composer(&text);
                    self.completion.visible = text.starts_with('/');
                    self.completion.selected = 0;
                }
            }
            // Ctrl+letter shortcuts that aren't bound are ignored. Ctrl+Alt is
            // AltGr on Windows and produces real characters (e.g. "/" on ABNT2).
            KeyCode::Char(_) if ctrl && !alt => {}
            KeyCode::Char(ch) => {
                self.history.clear_navigation();
                self.composer.insert(ch);
                self.completion.selected = 0;
                self.completion.visible = self.composer.text().starts_with('/');
            }
            _ => {}
        }
    }

    /// Large pastes become a "[Pasted text #N +K lines]" placeholder that is
    /// expanded back to the full text on submit.
    fn handle_paste(&mut self, text: String) {
        if self.confirmation.is_some() || self.overlay == Overlay::ModelPicker {
            return;
        }
        self.search = None;
        let text = text.replace("\r\n", "\n").replace('\r', "\n");
        let lines = text.matches('\n').count() + 1;
        self.history.clear_navigation();
        if lines > PASTE_PLACEHOLDER_LINES || text.chars().count() > PASTE_PLACEHOLDER_CHARS {
            let placeholder = format!("[Pasted text #{} +{lines} lines]", self.pastes.len() + 1);
            self.composer.insert_str(&placeholder);
            self.pastes.push((placeholder, text));
        } else {
            self.composer.insert_str(&text);
        }
        self.completion.visible = false;
    }

    fn clear_composer(&mut self) {
        self.composer.clear();
        self.pastes.clear();
        self.completion.selected = 0;
        self.completion.visible = false;
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

    fn replace_composer(&mut self, text: &str) {
        self.composer.clear();
        self.composer.insert_str(text);
    }

    fn submit(&mut self) {
        let text = self.composer.text();

        if text.trim().is_empty() {
            return;
        }

        self.history.push(&text);
        let trimmed = text.trim();
        // Submitting closes the shortcuts panel. This happens before the
        // command runs so commands that open a panel (/models) keep it open.
        self.overlay = Overlay::None;

        if let Some(command) = trimmed.strip_prefix('/') {
            let mut parts = command.trim().splitn(2, char::is_whitespace);
            let name = parts.next().unwrap_or("").to_lowercase();
            let arg = parts.next().unwrap_or("").trim().to_string();
            self.handle_command(&name, &arg, trimmed.to_string());
        } else if matches!(trimmed.to_lowercase().as_str(), "exit" | "quit") {
            self.running = false;
        } else {
            if self.run.is_some() {
                // Keep the draft so it can be sent once the current run ends.
                self.push(MessageRole::Muted, BUSY_MESSAGE.into());
                self.scroll.follow_end = true;
                return;
            }
            // The transcript shows what was typed (with paste placeholders);
            // the model receives the full pasted text.
            let prompt = expand_pastes(trimmed, &self.pastes);
            self.push(MessageRole::User, trimmed.to_string());
            let budget = self
                .orchestrator
                .as_ref()
                .map_or(4000, |o| o.config.max_tool_output_chars);
            let (prompt, attached) = attach_files(&prompt, &self.workspace, budget);
            if !attached.is_empty() {
                self.push(MessageRole::Muted, format!("Attached: {}", attached.join(", ")));
            }
            self.start_run(prompt);
        }

        self.clear_composer();
        self.scroll.follow_end = true;
    }

    fn handle_command(&mut self, command: &str, arg: &str, raw: String) {
        let needs_orchestrator = match command {
            "new" | "reset" | "clear" | "models" => true,
            "model" | "workspace" => !arg.is_empty(),
            _ => false,
        };
        if needs_orchestrator && self.orchestrator.is_none() {
            self.push(MessageRole::User, raw);
            self.push(MessageRole::Muted, BUSY_MESSAGE.into());
            return;
        }

        if command != "clear" {
            self.push(MessageRole::User, raw);
        }

        match command {
            "new" | "reset" => {
                self.orchestrator_mut().reset_history();
                self.messages.clear();
                self.overlay = Overlay::None;
                self.push(MessageRole::Success, "Conversation cleared.".into());
            }
            "clear" => {
                self.orchestrator_mut().reset_history();
                self.messages.clear();
                self.overlay = Overlay::None;
            }
            "quit" | "exit" | "q" => self.running = false,
            "help" => self.push(MessageRole::Help, String::new()),
            "models" => self.open_model_picker(),
            "stats" => match self.last_stats.clone() {
                Some(stats) => self.push(MessageRole::Muted, format!("[{stats}]")),
                None => self.push(MessageRole::Muted, "No stats yet.".into()),
            },
            "ls" | "tree" => {
                // /ls shows one level, /tree two; both accept a path and
                // --depth=N (also "--depth N", "-d N" or a bare number).
                let default_depth = if command == "ls" { 1 } else { 2 };
                match parse_tree_args(arg, default_depth) {
                    Ok((path, depth)) => {
                        let output = FilesystemSandbox::new(&self.workspace)
                            .and_then(|sandbox| sandbox.tree(&path, depth as i64, false, true));
                        match output {
                            Ok(tree) => {
                                if let Some(orchestrator) = self.orchestrator.as_mut() {
                                    orchestrator.add_context(&format!(
                                        "The user ran /{command} and Echo showed them:\n{tree}"
                                    ));
                                }
                                self.push(MessageRole::Plain, tree);
                            }
                            Err(error) => self.push(MessageRole::Error, format!("Error: {error}")),
                        }
                    }
                    Err(error) => self.push(
                        MessageRole::Usage,
                        format!("Usage: /{command} [path] [--depth=N]\n{error}"),
                    ),
                }
            }
            "model" => {
                if arg.is_empty() {
                    self.push(
                        MessageRole::Usage,
                        format!("Usage: /model <name>\nCurrent model: {}", self.model),
                    );
                } else {
                    match self.set_model(arg) {
                        Ok(()) => self.push(
                            MessageRole::Success,
                            format!("Model switched to: {}", self.model),
                        ),
                        Err(message) => self.push(MessageRole::Muted, message),
                    }
                }
            }
            "workspace" => {
                if arg.is_empty() {
                    self.push(
                        MessageRole::Usage,
                        format!(
                            "Usage: /workspace <path>\nCurrent workspace: {}",
                            self.workspace.display()
                        ),
                    );
                } else {
                    self.change_workspace(arg);
                }
            }
            _ => self.push(
                MessageRole::Error,
                format!("Unknown command: /{command}\nType /help to see available commands."),
            ),
        }
    }

    fn change_workspace(&mut self, arg: &str) {
        let path = expand_home(arg);
        let path = match std::fs::canonicalize(&path) {
            Ok(path) if path.is_dir() => display_path(path),
            _ => {
                self.push(
                    MessageRole::Error,
                    format!("Error: '{arg}' is not a valid directory."),
                );
                return;
            }
        };

        let mut config = self.orchestrator_mut().config.clone();
        config.workspace = path.clone();
        let mut tools = ToolRegistry::new(config.max_tool_output_chars);
        let _ = crate::tools::register_all(&mut tools, &path);
        let mut orchestrator = match EchoOrchestrator::new(config, tools) {
            Ok(orchestrator) => orchestrator,
            Err(error) => {
                self.push(
                    MessageRole::Error,
                    format!("Error initializing Ollama: {error}"),
                );
                return;
            }
        };
        orchestrator.tool_output_visible = self.verbose;
        let _ = orchestrator.config.save();
        self.orchestrator = Some(orchestrator);
        self.workspace = path;
        self.history.set_workspace(&self.workspace);
        self.push(
            MessageRole::Success,
            format!("Workspace changed to: {}", self.workspace.display()),
        );
        self.push(
            MessageRole::Muted,
            format!("[Tools reloaded -> {}]", self.workspace.display()),
        );
        self.start_warm_up();
    }

    fn open_model_picker(&mut self) {
        let client = self.orchestrator_mut().client.clone();
        let models = match client.list_models(Duration::from_secs(5)) {
            Ok(models) => models,
            Err(error) => {
                self.push(
                    MessageRole::Error,
                    format!("Could not list models from {}: {error}", self.ollama_url),
                );
                return;
            }
        };
        if models.is_empty() {
            self.push(
                MessageRole::Usage,
                "No models installed. Run: ollama pull <name>".into(),
            );
            return;
        }

        let current = normalize_model_name(&self.model);
        self.model_picker.selected = models
            .iter()
            .position(|model| normalize_model_name(&model.name) == current)
            .unwrap_or(0);
        self.model_picker.models = models;
        self.overlay = Overlay::ModelPicker;
        self.completion.visible = false;
        self.scroll.follow_end = true;
    }

    fn orchestrator_mut(&mut self) -> &mut EchoOrchestrator {
        self.orchestrator
            .as_mut()
            .expect("orchestrator is only taken while a run is in progress")
    }

    fn push(&mut self, role: MessageRole, content: String) {
        self.messages.push(Message { role, content });
    }

    fn set_model(&mut self, model: &str) -> Result<(), String> {
        let Some(orchestrator) = self.orchestrator.as_mut() else {
            return Err(BUSY_MESSAGE.into());
        };
        self.model = model.to_string();
        orchestrator.config.model = self.model.clone();
        let _ = orchestrator.config.save();
        self.start_warm_up();
        Ok(())
    }

    /// Preloads the model in the background so the first prompt doesn't pay
    /// for loading it and for evaluating the system prompt and tools. It is
    /// silent: a failure shows up when the user sends a prompt.
    fn start_warm_up(&mut self) {
        let Some(orchestrator) = self.orchestrator.as_mut() else {
            return;
        };
        let task = orchestrator.warm_up_task();
        let (sender, receiver) = mpsc::channel();
        thread::spawn(move || {
            let _ = task();
            let _ = sender.send(());
        });
        self.warm_up = Some(receiver);
    }

    fn start_run(&mut self, prompt: String) {
        let Some(mut orchestrator) = self.orchestrator.take() else {
            return;
        };
        let cancel = Arc::clone(&orchestrator.cancel);
        let (sender, receiver) = mpsc::channel();
        thread::spawn(move || {
            let events = sender.clone();
            let confirmations = sender.clone();
            let result = orchestrator.run(
                prompt,
                None,
                |event| {
                    let _ = events.send(WorkerMessage::Event(event));
                },
                // Blocks the worker until the user answers in the UI.
                |tool_name, arguments| {
                    let (reply, answer) = mpsc::channel();
                    let request = WorkerMessage::Confirm {
                        tool_name: tool_name.to_string(),
                        arguments: arguments.clone(),
                        reply,
                    };
                    confirmations.send(request).is_ok() && answer.recv().unwrap_or(false)
                },
            );
            let _ = sender.send(WorkerMessage::Finished(Box::new(orchestrator), result));
        });
        self.run = Some(PendingRun {
            events: receiver,
            cancel,
            reply: None,
        });
        self.waiting_since = Some(Instant::now());
    }

    /// Returns whether there was a run to cancel.
    fn cancel_run(&mut self) -> bool {
        let Some(run) = self.run.as_ref() else {
            return false;
        };
        run.cancel.store(true, Ordering::Relaxed);
        if self.confirmation.is_some() {
            self.resolve_confirmation(false);
        }
        true
    }

    /// Drains events from the worker threads without blocking the UI.
    fn poll_background(&mut self) {
        if let Some(receiver) = self.warm_up.as_ref() {
            if !matches!(receiver.try_recv(), Err(TryRecvError::Empty)) {
                self.warm_up = None;
            }
        }

        loop {
            let Some(run) = self.run.as_ref() else {
                return;
            };
            match run.events.try_recv() {
                Ok(WorkerMessage::Event(event)) => self.apply_event(event),
                Ok(WorkerMessage::Confirm {
                    tool_name,
                    arguments,
                    reply,
                }) => {
                    self.waiting_since = None;
                    self.confirmation = Some(Confirmation {
                        details: confirmation_details(&tool_name, &arguments, &self.workspace),
                        choice: None,
                        reply,
                    });
                    self.scroll.follow_end = true;
                }
                Ok(WorkerMessage::Finished(orchestrator, result)) => {
                    self.finish_run(*orchestrator, result);
                    return;
                }
                Err(TryRecvError::Empty) => return,
                Err(TryRecvError::Disconnected) => {
                    // The worker panicked and took the orchestrator with it.
                    self.run = None;
                    self.waiting_since = None;
                    self.push(
                        MessageRole::Error,
                        "[Error] The request failed unexpectedly. Restart Echo.".into(),
                    );
                    return;
                }
            }
        }
    }

    fn apply_event(&mut self, event: OrchestratorEvent) {
        let Some(run) = self.run.as_mut() else {
            return;
        };
        match event {
            OrchestratorEvent::IterationStarted => {
                run.reply = None;
                self.waiting_since = Some(Instant::now());
            }
            OrchestratorEvent::ContentDelta(delta) => {
                self.waiting_since = None;
                let index = *run.reply.get_or_insert_with(|| {
                    self.messages.push(Message {
                        role: MessageRole::Assistant,
                        content: String::new(),
                    });
                    self.messages.len() - 1
                });
                self.messages[index].content.push_str(&delta);
            }
            OrchestratorEvent::ToolCallReceived {
                tool_name,
                arguments,
            } => {
                self.waiting_since = None;
                run.reply = None;
                if self.verbose {
                    self.push(
                        MessageRole::Tool,
                        format!("[Tool Call]\n{tool_name}({})", pretty_arguments(&arguments)),
                    );
                }
            }
            OrchestratorEvent::ToolExecuted(record) => {
                if self.verbose {
                    let (role, content) = tool_result_message(&record);
                    self.push(role, content);
                }
            }
            OrchestratorEvent::EmptyResponseRetry => {}
        }
    }

    fn finish_run(&mut self, orchestrator: EchoOrchestrator, result: OrchestratorResult) {
        self.orchestrator = Some(orchestrator);
        self.waiting_since = None;
        if let Some(confirmation) = self.confirmation.take() {
            let _ = confirmation.reply.send(false);
        }
        let Some(run) = self.run.take() else {
            return;
        };

        let interrupted = run.cancel.load(Ordering::Relaxed)
            && result.stopped_reason == RunStopReason::UserCancelled;
        match (result.stopped_reason, &result.error_message) {
            (RunStopReason::Error, Some(error)) => {
                self.push(MessageRole::Error, format!("[Error] {error}"))
            }
            _ if interrupted => self.push(MessageRole::Muted, "[Interrupted]".into()),
            _ => match run.reply {
                Some(index) => self.messages[index].content = result.final_response.clone(),
                None => self.push(MessageRole::Assistant, result.final_response.clone()),
            },
        }

        let stats = format_stats(&result);
        if self.verbose {
            self.push(MessageRole::Muted, format!("[{stats}]"));
        }
        self.last_stats = Some(stats);
    }

    /// Banner and divider left on the terminal after Echo exits.
    pub fn exit_screen(&self, width: u16) -> String {
        let mut out = ui::banner_lines(self)
            .into_iter()
            .map(|line| line.trim_end().to_string())
            .collect::<Vec<_>>()
            .join("\n");
        out.push('\n');
        out.push_str(&"─".repeat(usize::from(width.max(20))));
        out.push_str("\n\nExiting...\n");
        out
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

    pub fn model_picker_rows(&self) -> usize {
        self.model_picker.models.len().clamp(1, MODEL_PICKER_ROWS)
    }
}

/// Detects a paste delivered as a burst of key presses (Windows consoles do
/// not send bracketed-paste events). Typing never queues this many printable
/// keys, or a line break followed by more text, within one frame.
fn paste_burst(batch: &[Event]) -> Option<String> {
    let mut text = String::new();
    for event in batch {
        let Event::Key(key) = event else {
            return None;
        };
        if key.kind == KeyEventKind::Release {
            continue;
        }
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL)
            && !key.modifiers.contains(KeyModifiers::ALT);
        match key.code {
            KeyCode::Char(ch) if !ctrl => text.push(ch),
            KeyCode::Enter => text.push('\n'),
            KeyCode::Tab => text.push('\t'),
            _ => return None,
        }
    }
    let count = text.chars().count();
    let has_inner_break = text.trim_end_matches('\n').contains('\n');
    (count >= 8 || (has_inner_break && count >= 3)).then_some(text)
}

/// Maximum depth accepted by /ls and /tree (the tree tool's own limit).
const MAX_TREE_DEPTH: usize = 6;

/// Parses `[path] [--depth=N | --depth N | -d N | N]` for /ls and /tree.
fn parse_tree_args(arg: &str, default_depth: usize) -> Result<(String, usize), String> {
    let mut path: Option<String> = None;
    let mut depth = default_depth;
    let mut words = arg.split_whitespace();
    let parse_depth = |value: Option<&str>| -> Result<usize, String> {
        let value = value.ok_or("--depth needs a number")?;
        match value.parse::<usize>() {
            Ok(n) if (1..=MAX_TREE_DEPTH).contains(&n) => Ok(n),
            _ => Err(format!("Invalid depth: {value} (use 1-{MAX_TREE_DEPTH})")),
        }
    };

    while let Some(word) = words.next() {
        if let Some(value) = word.strip_prefix("--depth=") {
            depth = parse_depth(Some(value))?;
        } else if word == "--depth" || word == "-d" {
            depth = parse_depth(words.next())?;
        } else if let Some(value) = word.strip_prefix("-d").filter(|v| !v.is_empty()) {
            depth = parse_depth(Some(value))?;
        } else if word.starts_with('-') {
            return Err(format!("Unknown option: {word}"));
        } else if word.chars().all(|c| c.is_ascii_digit()) {
            // Bare number, as in the old "/tree 3".
            depth = parse_depth(Some(word))?;
        } else if path.is_none() {
            path = Some(word.to_string());
        } else {
            return Err(format!("Unexpected argument: {word}"));
        }
    }
    Ok((path.unwrap_or_else(|| ".".into()), depth))
}

/// Appends the content of every `@path` in the prompt that names a file in
/// the workspace (read with line numbers, like read_file). Returns the new
/// prompt and the attached paths. Unknown `@words` are left alone.
pub fn attach_files(prompt: &str, workspace: &Path, budget: usize) -> (String, Vec<String>) {
    let Ok(sandbox) = FilesystemSandbox::new(workspace) else {
        return (prompt.to_string(), Vec::new());
    };
    let mut attached: Vec<String> = Vec::new();
    let mut text = prompt.to_string();
    for word in prompt.split_whitespace() {
        let Some(name) = word.strip_prefix('@').filter(|name| !name.is_empty()) else {
            continue;
        };
        // "@Main.java," or "(@a.txt)" still attach the file.
        let trimmed = name.trim_end_matches(['.', ',', ';', ':', '!', '?', ')', '"', '\'']);
        for candidate in [name, trimmed] {
            if attached.iter().any(|path| path == candidate)
                || !workspace.join(candidate).is_file()
            {
                continue;
            }
            if let Ok(content) = sandbox.read_file(candidate, 1, budget) {
                text.push_str(&format!("\n\n[Attached file: {candidate}]\n{content}"));
                attached.push(candidate.to_string());
                break;
            }
        }
    }
    (text, attached)
}

fn expand_pastes(text: &str, pastes: &[(String, String)]) -> String {
    pastes
        .iter()
        .fold(text.to_string(), |text, (placeholder, original)| {
            text.replace(placeholder, original)
        })
}

fn expand_home(path: &str) -> PathBuf {
    match path.strip_prefix('~') {
        Some(rest) if rest.is_empty() || rest.starts_with(['/', '\\']) => {
            let home = std::env::var_os("USERPROFILE")
                .or_else(|| std::env::var_os("HOME"))
                .map(PathBuf::from)
                .unwrap_or_default();
            home.join(rest.trim_start_matches(['/', '\\']))
        }
        _ => PathBuf::from(path),
    }
}

/// Tool arguments as indented JSON, like the Python transcript.
pub fn pretty_arguments(raw: &str) -> String {
    serde_json::from_str::<Value>(raw)
        .ok()
        .and_then(|value| serde_json::to_string_pretty(&value).ok())
        .unwrap_or_else(|| raw.trim().to_string())
}

/// "[Tool Executed (OK) in 0.001s] -> result", with long results cut to
/// 15 lines. `tree` output is shown in full under the header.
pub fn tool_result_message(record: &ToolExecutionRecord) -> (MessageRole, String) {
    let status = if record.success { "OK" } else { "FAILED" };
    let header = format!(
        "[Tool Executed ({status}) in {:.3}s]",
        record.duration_seconds
    );
    if record.tool_name == "tree" {
        return (MessageRole::ToolResult, format!("{header}\n{}", record.result));
    }
    let lines = record.result.lines().collect::<Vec<_>>();
    let mut result = lines
        .iter()
        .take(TOOL_RESULT_LINES)
        .copied()
        .collect::<Vec<_>>()
        .join("\n");
    if lines.len() > TOOL_RESULT_LINES {
        result.push_str(&format!(
            "\n... (+{} lines hidden)",
            lines.len() - TOOL_RESULT_LINES
        ));
    }
    (MessageRole::Tool, format!("{header} -> {result}"))
}

/// apt-style summary of what a file-changing tool is about to do.
pub fn confirmation_details(tool_name: &str, arguments: &Value, workspace: &Path) -> String {
    let path_text = arguments
        .get("path")
        .and_then(Value::as_str)
        .unwrap_or("?");
    let text_len = |key: &str| {
        arguments
            .get(key)
            .and_then(Value::as_str)
            .map_or(0, |text| text.len() as i64)
    };
    let disk_line = |delta: i64| match delta {
        d if d > 0 => format!(
            "After this operation, {} of additional disk space will be used.",
            format_bytes(d as u64)
        ),
        d if d < 0 => format!(
            "After this operation, {} of disk space will be freed.",
            format_bytes(d.unsigned_abs())
        ),
        _ => "After this operation, no additional disk space will be used.".into(),
    };

    let text = |key: &str| arguments.get(key).and_then(Value::as_str).unwrap_or("");

    match tool_name {
        "write_file" => {
            let size = text_len("content");
            let path = workspace.join(path_text);
            let summary = match std::fs::metadata(&path) {
                Ok(meta) if meta.is_file() => format!(
                    "The following file will be modified:\n{path_text}\n0 created, 1 modified, 0 deleted.\n{}",
                    disk_line(size - meta.len() as i64)
                ),
                _ => format!(
                    "The following file will be created:\n{path_text}\n0 modified, 1 created, 0 deleted.\n{}",
                    disk_line(size)
                ),
            };
            with_preview(summary, &[('+', text("content"), PREVIEW_LINES, Some(1))])
        }
        "edit_file" => {
            // Same normalization as the tool: drop copied "N | " prefixes,
            // then find where the replaced text starts in the file.
            let file = std::fs::read_to_string(workspace.join(path_text)).unwrap_or_default();
            let (old, new) = match strip_line_numbers(text("old_text")) {
                Some(old) if !file.contains(text("old_text")) && file.contains(&old) => {
                    let new = strip_line_numbers(text("new_text"));
                    (old, new.unwrap_or_else(|| text("new_text").to_string()))
                }
                _ => (text("old_text").to_string(), text("new_text").to_string()),
            };
            let start = file.find(&old).filter(|_| !old.is_empty()).map(|at| line_of(&file, at));
            with_preview(
                format!(
                    "The following file will be modified:\n{path_text}\n0 created, 1 modified, 0 deleted.\n{}",
                    disk_line(new.len() as i64 - old.len() as i64)
                ),
                &[
                    ('-', &old, PREVIEW_LINES / 2, start),
                    ('+', &new, PREVIEW_LINES / 2, start),
                ],
            )
        }
        "delete_file" => {
            let size = std::fs::metadata(workspace.join(path_text)).map_or(0, |m| m.len());
            format!(
                "The following file will be deleted:\n{path_text}\n0 modified, 0 created, 1 deleted.\nAfter this operation, {} of disk space will be freed.",
                format_bytes(size)
            )
        }
        "move_file" => format!(
            "The following file will be moved:\n{} -> {}",
            text("source"),
            text("destination")
        ),
        "run_command" => {
            let timeout = crate::tools::shell::timeout_from(
                arguments.as_object().unwrap_or(&serde_json::Map::new()),
            );
            format!(
                "The following command will be run:\n{}\nIn {} with {}, stopped after {}s.",
                text("command"),
                workspace.display(),
                crate::tools::shell::shell_name(),
                timeout.as_secs()
            )
        }
        _ => format!("The following operation will be performed:\n{tool_name}"),
    }
}

/// Lines of the changed text shown under a confirmation summary.
const PREVIEW_LINES: usize = 8;
const PREVIEW_LINE_CHARS: usize = 120;
/// Separates the line number from the text in previews ("+ 12 │ code").
pub const PREVIEW_GUTTER: &str = " │ ";

/// Appends "+ added" / "- removed" lines, each block cut to `limit` lines.
/// Blocks with a starting line number show it in front of each line.
fn with_preview(mut summary: String, blocks: &[(char, &str, usize, Option<usize>)]) -> String {
    let width = blocks
        .iter()
        .filter_map(|(_, text, limit, start)| {
            start.map(|start| start + text.lines().count().min(*limit).saturating_sub(1))
        })
        .max()
        .map_or(0, |last| last.to_string().len());

    for (marker, text, limit, start) in blocks {
        let lines = text.lines().collect::<Vec<_>>();
        for (index, line) in lines.iter().take(*limit).enumerate() {
            let shown = line.chars().take(PREVIEW_LINE_CHARS).collect::<String>();
            match start {
                Some(start) => summary.push_str(&format!(
                    "\n{marker} {:>width$}{PREVIEW_GUTTER}{shown}",
                    start + index
                )),
                None => summary.push_str(&format!("\n{marker} {shown}")),
            }
        }
        if lines.len() > *limit {
            summary.push_str(&format!("\n  ... (+{} more lines)", lines.len() - limit));
        }
    }
    summary
}

/// How each part of a confirmation request is colored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfirmTone {
    Header,
    Path,
    Created,
    Modified,
    Deleted,
    Added,
    Removed,
    Muted,
    Text,
    Yes,
    No,
}

/// Splits line `index` of a confirmation request (the text built by
/// `confirmation_details`, optionally followed by the question and answer)
/// into colored segments. Shared by the TUI and one-shot mode.
pub fn confirmation_segments(index: usize, line: &str) -> Vec<(ConfirmTone, String)> {
    if let Some(answer) = line.strip_prefix(CONFIRM_QUESTION) {
        let mut segments = vec![(ConfirmTone::Text, CONFIRM_QUESTION.to_string())];
        match answer {
            "y" => segments.push((ConfirmTone::Yes, answer.into())),
            "n" => segments.push((ConfirmTone::No, answer.into())),
            "" => {}
            other => segments.push((ConfirmTone::Text, other.into())),
        }
        return segments;
    }
    let whole = |tone| vec![(tone, line.to_string())];
    match index {
        0 => return whole(ConfirmTone::Header),
        1 => return whole(ConfirmTone::Path),
        _ => {}
    }
    let change = if line.starts_with("+ ") {
        Some(ConfirmTone::Added)
    } else if line.starts_with("- ") {
        Some(ConfirmTone::Removed)
    } else {
        None
    };
    if let Some(tone) = change {
        // "+ 12 │ code": the line number is muted, the code keeps the color.
        let body = &line[2..];
        let number = body.trim_start();
        let digits = number.chars().take_while(char::is_ascii_digit).count();
        if digits > 0 && number[digits..].starts_with(PREVIEW_GUTTER) {
            let split = 2 + (body.len() - number.len()) + digits + PREVIEW_GUTTER.len();
            return vec![
                (tone, line[..2].to_string()),
                (ConfirmTone::Muted, line[2..split].to_string()),
                (tone, line[split..].to_string()),
            ];
        }
        return whole(tone);
    }
    if let Some(counts) = line.strip_suffix(" deleted.") {
        // "0 modified, 1 created, 0 deleted." highlights the non-zero part.
        let parts = format!("{counts} deleted");
        let mut segments = Vec::new();
        for (i, part) in parts.split(", ").enumerate() {
            if i > 0 {
                segments.push((ConfirmTone::Muted, ", ".to_string()));
            }
            let tone = match part.split_once(' ') {
                Some((count, _)) if count == "0" => ConfirmTone::Muted,
                Some((_, "created")) => ConfirmTone::Created,
                Some((_, "modified")) => ConfirmTone::Modified,
                Some((_, "deleted")) => ConfirmTone::Deleted,
                _ => ConfirmTone::Muted,
            };
            segments.push((tone, part.to_string()));
        }
        segments.push((ConfirmTone::Muted, ".".to_string()));
        return segments;
    }
    whole(ConfirmTone::Muted)
}

fn format_bytes(value: u64) -> String {
    let mut size = value as f64;
    for unit in ["B", "KB", "MB", "GB"] {
        if size < 1024.0 || unit == "GB" {
            return if unit == "B" {
                format!("{value} B")
            } else {
                format!("{size:.1} {unit}")
            };
        }
        size /= 1024.0;
    }
    format!("{value} B")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(code: KeyCode) -> Event {
        Event::Key(KeyEvent::new(code, KeyModifiers::NONE))
    }

    #[test]
    fn clear_command_removes_existing_history() {
        let mut app = EchoApp::new();
        app.messages.push(Message {
            role: MessageRole::User,
            content: "hello".into(),
        });
        app.messages.push(Message {
            role: MessageRole::Assistant,
            content: "response".into(),
        });

        app.handle_command("clear", "", "/clear".into());

        assert!(app.messages.is_empty());
        assert_eq!(app.overlay, Overlay::None);
        assert!(app.running);
    }

    #[test]
    fn new_command_clears_conversation_and_reports_it() {
        let mut app = EchoApp::new();
        app.messages.push(Message {
            role: MessageRole::User,
            content: "old".into(),
        });

        app.handle_command("new", "", "/new".into());

        assert_eq!(app.messages.len(), 1);
        assert_eq!(app.messages[0].content, "Conversation cleared.");
        assert_eq!(app.messages[0].role, MessageRole::Success);
    }

    #[test]
    fn reset_command_is_new_alias() {
        let mut app = EchoApp::new();
        app.messages.push(Message {
            role: MessageRole::User,
            content: "old".into(),
        });

        app.handle_command("reset", "", "/reset".into());

        assert_eq!(app.messages.len(), 1);
        assert_eq!(app.messages[0].content, "Conversation cleared.");
    }

    #[test]
    fn unknown_command_points_to_help() {
        let mut app = EchoApp::new();

        app.handle_command("does-not-exist", "", "/does-not-exist".into());

        assert!(app.messages.last().unwrap().content.contains("Type /help"));
    }

    #[test]
    fn exit_command_stops_application() {
        let mut app = EchoApp::new();

        app.handle_command("exit", "", "/exit".into());

        assert!(!app.running);
    }

    #[test]
    fn quit_command_is_exit_alias() {
        let mut app = EchoApp::new();

        app.handle_command("quit", "", "/quit".into());

        assert!(!app.running);
    }

    #[test]
    fn uppercase_command_is_normalized_on_submit() {
        let mut app = EchoApp::new();
        for ch in "/EXIT".chars() {
            app.composer.insert(ch);
        }

        app.submit();

        assert!(!app.running);
    }

    #[test]
    fn models_command_keeps_its_picker_open() {
        let mut app = EchoApp::new();
        app.history = History::new();
        app.composer.insert_str("/models");
        app.submit();
        // Needs a running Ollama for the list; without it an error is shown.
        let error = app.messages.last().is_some_and(|m| m.content.contains("Could not list models"));
        assert!(app.overlay == Overlay::ModelPicker || error);
    }

    #[test]
    fn enter_runs_the_highlighted_completion() {
        let mut app = EchoApp::new();
        app.history = History::new();
        for ch in "/st".chars() {
            app.handle_key(KeyEvent::new(KeyCode::Char(ch), KeyModifiers::NONE));
        }
        app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        assert_eq!(app.messages[0].content, "/stats");
        assert_eq!(app.messages[1].content, "No stats yet.");

        // Commands that need an argument wait for it.
        for ch in "/mo".chars() {
            app.handle_key(KeyEvent::new(KeyCode::Char(ch), KeyModifiers::NONE));
        }
        app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        assert_eq!(app.composer.text(), "/model ");
    }

    #[test]
    fn mouse_drag_selects_and_requests_a_copy() {
        let mut app = EchoApp::new();
        let mouse = |kind, column, row| MouseEvent {
            kind,
            column,
            row,
            modifiers: KeyModifiers::NONE,
        };
        app.handle_mouse(mouse(MouseEventKind::Down(MouseButton::Left), 2, 10));
        app.handle_mouse(mouse(MouseEventKind::Drag(MouseButton::Left), 8, 11));
        app.handle_mouse(mouse(MouseEventKind::Up(MouseButton::Left), 9, 11));
        let selection = app.selection.unwrap();
        assert_eq!((selection.anchor, selection.head), ((2, 10), (9, 11)));
        assert!(selection.copy && !selection.dragging);

        // A click without dragging, the wheel or a key press clears it.
        app.handle_mouse(mouse(MouseEventKind::Down(MouseButton::Left), 3, 3));
        app.handle_mouse(mouse(MouseEventKind::Up(MouseButton::Left), 3, 3));
        assert!(app.selection.is_none());
    }

    #[test]
    fn tree_arguments() {
        assert_eq!(parse_tree_args("", 2), Ok((".".into(), 2)));
        assert_eq!(parse_tree_args("--depth=3", 2), Ok((".".into(), 3)));
        assert_eq!(parse_tree_args("src --depth 4", 2), Ok(("src".into(), 4)));
        assert_eq!(parse_tree_args("-d 5 src", 1), Ok(("src".into(), 5)));
        assert_eq!(parse_tree_args("-d2", 1), Ok((".".into(), 2)));
        assert_eq!(parse_tree_args("3", 2), Ok((".".into(), 3)));
        assert!(parse_tree_args("--depth=0", 2).is_err());
        assert!(parse_tree_args("--depth=9", 2).is_err());
        assert!(parse_tree_args("--depth", 2).is_err());
        assert!(parse_tree_args("--all", 2).is_err());
        assert!(parse_tree_args("a b", 2).is_err());
    }

    #[test]
    fn ls_and_tree_render_the_workspace_as_a_tree() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("src/deep")).unwrap();
        std::fs::write(dir.path().join("src/deep/x.rs"), "x").unwrap();
        std::fs::write(dir.path().join("Main.java"), "class Main {}").unwrap();
        let mut app = EchoApp::new();
        app.workspace = dir.path().to_path_buf();

        app.handle_command("ls", "", "/ls".into());
        let ls = app.messages.last().unwrap().content.clone();
        assert!(ls.contains("├── src/") && ls.contains("└── Main.java (13 bytes)"), "{ls}");
        assert!(!ls.contains("deep"));

        app.handle_command("tree", "--depth=3", "/tree --depth=3".into());
        let tree = app.messages.last().unwrap().content.clone();
        assert!(tree.contains("deep/") && tree.contains("x.rs"), "{tree}");

        // The model gets to see what the user saw.
        let history = &app.orchestrator.as_ref().unwrap().history;
        let note = history.last().unwrap()["content"].as_str().unwrap();
        assert!(note.starts_with("[Echo] The user ran /tree") && note.contains("x.rs"), "{note}");
    }

    #[test]
    fn at_mentions_attach_workspace_files() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("Main.java"), "class Main {}\n").unwrap();

        let (prompt, attached) = attach_files(
            "Explain @Main.java, and ping user@example.com about @missing.txt",
            dir.path(),
            4000,
        );
        assert_eq!(attached, vec!["Main.java".to_string()]);
        assert!(prompt.ends_with("[Attached file: Main.java]\n1 | class Main {}\n"), "{prompt}");
        assert!(!prompt.contains("[Attached file: missing.txt]"));
    }

    #[test]
    fn plain_exit_word_quits() {
        let mut app = EchoApp::new();
        app.composer.insert_str("quit");
        app.submit();
        assert!(!app.running);
    }

    #[test]
    fn key_release_events_are_ignored() {
        let mut app = EchoApp::new();
        let mut press = KeyEvent::new(KeyCode::Char('a'), KeyModifiers::NONE);
        app.handle_key(press);
        press.kind = KeyEventKind::Release;
        app.handle_key(press);
        assert_eq!(app.composer.text(), "a");
    }

    #[test]
    fn ctrl_c_clears_input_before_exiting() {
        let mut app = EchoApp::new();
        app.composer.insert_str("draft");
        let ctrl_c = KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL);

        app.handle_key(ctrl_c);
        assert!(app.composer.is_empty());
        assert!(app.running);

        app.handle_key(ctrl_c);
        assert!(!app.running);
    }

    #[test]
    fn ctrl_j_inserts_a_line_break() {
        let mut app = EchoApp::new();
        app.composer.insert('a');
        app.handle_key(KeyEvent::new(KeyCode::Char('j'), KeyModifiers::CONTROL));
        app.composer.insert('b');
        assert_eq!(app.composer.text(), "a\nb");
    }

    #[test]
    fn altgr_characters_are_typed() {
        let mut app = EchoApp::new();
        app.handle_key(KeyEvent::new(
            KeyCode::Char('/'),
            KeyModifiers::CONTROL | KeyModifiers::ALT,
        ));
        assert_eq!(app.composer.text(), "/");
    }

    #[test]
    fn large_paste_becomes_a_placeholder_and_expands_on_submit() {
        let mut app = EchoApp::new();
        let pasted = (1..=7).map(|n| format!("line {n}")).collect::<Vec<_>>().join("\n");
        app.handle_paste(pasted.clone());

        assert_eq!(app.composer.text(), "[Pasted text #1 +7 lines]");
        assert_eq!(expand_pastes(&app.composer.text(), &app.pastes), pasted);
    }

    #[test]
    fn key_burst_is_treated_as_a_paste() {
        let burst = "ab\ncd"
            .chars()
            .map(|ch| key(if ch == '\n' { KeyCode::Enter } else { KeyCode::Char(ch) }))
            .collect::<Vec<_>>();
        assert_eq!(paste_burst(&burst).as_deref(), Some("ab\ncd"));

        let typing = vec![key(KeyCode::Char('a')), key(KeyCode::Char('b'))];
        assert!(paste_burst(&typing).is_none());
        assert!(paste_burst(&[key(KeyCode::Char('a')), key(KeyCode::Enter)]).is_none());
    }

    #[test]
    fn reverse_search_fills_the_composer() {
        let mut app = EchoApp::new();
        app.history = History::new();
        app.history.push("read notes.txt");
        app.history.push("list files");

        app.handle_key(KeyEvent::new(KeyCode::Char('r'), KeyModifiers::CONTROL));
        for ch in "notes".chars() {
            app.handle_key(KeyEvent::new(KeyCode::Char(ch), KeyModifiers::NONE));
        }
        app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));

        assert!(app.search.is_none());
        assert_eq!(app.composer.text(), "read notes.txt");
    }

    #[test]
    fn confirmation_defaults_to_yes_and_is_recorded() {
        let mut app = EchoApp::new();
        let (reply, answer) = mpsc::channel();
        app.confirmation = Some(Confirmation {
            details: "The following file will be created:\na.txt".into(),
            choice: None,
            reply,
        });

        app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));

        assert_eq!(answer.recv().unwrap(), true);
        assert!(app.confirmation.is_none());
        assert!(app.messages.last().unwrap().content.ends_with("[Y/n] y"));
    }

    #[test]
    fn confirmation_can_be_denied() {
        let mut app = EchoApp::new();
        let (reply, answer) = mpsc::channel();
        app.confirmation = Some(Confirmation {
            details: String::new(),
            choice: None,
            reply,
        });

        app.handle_key(KeyEvent::new(KeyCode::Char('n'), KeyModifiers::NONE));
        app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));

        assert_eq!(answer.recv().unwrap(), false);
    }

    #[test]
    fn confirmation_details_follow_apt_style() {
        let dir = std::env::temp_dir();
        let details = confirmation_details(
            "write_file",
            &serde_json::json!({"path": "echo-new-file-for-test.txt", "content": "Teste de confirmação."}),
            &dir,
        );
        assert_eq!(
            details,
            "The following file will be created:\necho-new-file-for-test.txt\n0 modified, 1 created, 0 deleted.\nAfter this operation, 23 B of additional disk space will be used.\n+ 1 │ Teste de confirmação."
        );
    }

    #[test]
    fn edit_preview_shows_removed_and_added_lines() {
        let details = confirmation_details(
            "edit_file",
            &serde_json::json!({"path": "a.txt", "old_text": "old", "new_text": "new\nmore"}),
            Path::new("."),
        );
        assert!(details.ends_with("\n- old\n+ new\n+ more"));
    }

    #[test]
    fn edit_preview_numbers_lines_from_the_file() {
        let dir = tempfile::tempdir().unwrap();
        let lines = (1..=12).map(|n| format!("line {n}")).collect::<Vec<_>>().join("\n");
        std::fs::write(dir.path().join("a.txt"), lines).unwrap();
        let details = confirmation_details(
            "edit_file",
            // Copied with read_file's prefixes; still located at line 9.
            &serde_json::json!({"path": "a.txt", "old_text": " 9 | line 9\n10 | line 10", "new_text": " 9 | nine"}),
            dir.path(),
        );
        assert!(
            details.ends_with("\n-  9 │ line 9\n- 10 │ line 10\n+  9 │ nine"),
            "{details}"
        );
    }

    #[test]
    fn long_previews_are_cut() {
        let content = (1..=20).map(|n| n.to_string()).collect::<Vec<_>>().join("\n");
        let details = with_preview(String::from("x"), &[('+', &content, 8, None)]);
        assert!(details.ends_with("\n+ 8\n  ... (+12 more lines)"));
    }

    #[test]
    fn confirmation_lines_are_split_into_colored_segments() {
        let counts = confirmation_segments(2, "0 modified, 1 created, 0 deleted.");
        assert!(counts.contains(&(ConfirmTone::Created, "1 created".into())));
        assert!(counts.contains(&(ConfirmTone::Muted, "0 modified".into())));
        let joined = counts.iter().map(|(_, text)| text.as_str()).collect::<String>();
        assert_eq!(joined, "0 modified, 1 created, 0 deleted.");

        assert_eq!(confirmation_segments(0, "The following file will be created:")[0].0, ConfirmTone::Header);
        assert_eq!(confirmation_segments(1, "a.txt")[0].0, ConfirmTone::Path);
        assert_eq!(confirmation_segments(5, "+ added")[0].0, ConfirmTone::Added);
        assert_eq!(confirmation_segments(5, "- removed")[0].0, ConfirmTone::Removed);
        let numbered = confirmation_segments(5, "+ 12 │ let x = 1;");
        assert_eq!(numbered[1], (ConfirmTone::Muted, "12 │ ".to_string()));
        assert_eq!(numbered[2], (ConfirmTone::Added, "let x = 1;".to_string()));
        let answer = confirmation_segments(6, &format!("{CONFIRM_QUESTION}n"));
        assert_eq!(answer.last().unwrap(), &(ConfirmTone::No, "n".to_string()));
    }

    #[test]
    fn long_tool_results_are_cut_in_the_transcript() {
        let record = ToolExecutionRecord {
            call_id: "1".into(),
            tool_name: "read_file".into(),
            arguments: Default::default(),
            raw_arguments: String::new(),
            result: (1..=20).map(|n| n.to_string()).collect::<Vec<_>>().join("\n"),
            success: true,
            duration_seconds: 0.001,
            error: None,
        };
        let (role, content) = tool_result_message(&record);
        assert_eq!(role, MessageRole::Tool);
        assert!(content.starts_with("[Tool Executed (OK) in 0.001s] -> 1\n"));
        assert!(content.ends_with("... (+5 lines hidden)"));
    }
}
