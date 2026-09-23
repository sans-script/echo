"""Command-line interface for Echo."""

import argparse
import json
import re
import shutil
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.filters import has_completions
from prompt_toolkit.layout import ConditionalContainer
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.layout.margins import Margin
from prompt_toolkit.layout import Window
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.cursor_shapes import CursorShape
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.history import History
from prompt_toolkit import prompt
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import (
    Float,
    FloatContainer,
    HSplit,
    Layout,
    VSplit,
    Window,
)
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.output.defaults import create_output
from prompt_toolkit.output.vt100 import Vt100_Output
from prompt_toolkit.renderer import CPR_Support
from prompt_toolkit.styles import Style

from .client import OllamaClient
from .config import EchoConfig
from .orchestrator import EchoOrchestrator


# ============================================================
# ANSI colors
# ============================================================

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"

WHITE = "\033[97m"
GRAY = "\033[90m"
GREEN = "\033[32m"
RED = "\033[31m"
CYAN = "\033[36m"
YELLOW = "\033[33m"


# ============================================================
# Cursor control (ANSI)
# ============================================================

# DECTCEM: hide cursor (used during model streaming).
CURSOR_HIDE = "\033[?25l"

# Restore cursor to visible + default shape when leaving the REPL.
CURSOR_RESTORE = "\033[?25h\033[0 q"


# ============================================================
# Echo logo
# ============================================================

ECHO_LOGO = [
    "         ::::::     ",
    "      :::     ::    ",
    "     :::     :::    ",
    "    ::::::::::      ",
    "    :::             ",
    "    :::        :    ",
    "    :::      :::    ",
    "      :::::::       ",
]


# ============================================================
# Loading animation
# ============================================================

SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


class Spinner:
    """
    Loading animation used while Echo is waiting for Ollama.
    """

    def __init__(self, message: str = "Loading"):
        self.message = message
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._frame = 0

    def start(self, message: Optional[str] = None):
        with self._lock:
            if message is not None:
                self.message = message

            if self._thread and self._thread.is_alive():
                return

            self._stop_event.clear()
            self._frame = 0
            self._thread = threading.Thread(target=self._spin, daemon=True)
            self._thread.start()

    def _render_loading(self, frame: int) -> str:
        text = f"{self.message}..."
        if not text:
            return ""

        position = frame % len(text)
        rendered = []

        for index, char in enumerate(text):
            distance = abs(index - position)
            if distance == 0:
                rendered.append(f"{WHITE}{BOLD}{char}{RESET}")
            elif distance == 1:
                rendered.append(f"{WHITE}{char}{RESET}")
            elif distance == 2:
                rendered.append(f"\033[37m{char}{RESET}")
            else:
                rendered.append(f"{GRAY}{char}{RESET}")

        return "".join(rendered)

    def _spin(self):
        while not self._stop_event.is_set():
            spinner_char = SPINNER_FRAMES[self._frame % len(SPINNER_FRAMES)]
            loading = self._render_loading(self._frame)

            sys.stdout.write(f"\r{GRAY}{spinner_char}{RESET} {loading}")
            sys.stdout.flush()

            self._frame += 1
            time.sleep(0.14)

        clear_length = len(self.message) + 12
        sys.stdout.write("\r" + (" " * clear_length) + "\r")
        sys.stdout.flush()

    def stop(self):
        with self._lock:
            self._stop_event.set()
            thread = self._thread

        if thread:
            thread.join(timeout=1.0)
        self._thread = None


class SpacesMargin(Margin):
    """Left margin que só escreve N espaços."""
    def __init__(self, width: int = 2):
        self.width = width

    def get_width(self, get_ui_content):
        return self.width

    def create_margin(self, window_render_info, width, height):
        # Retorna uma lista de fragments, uma por linha visível.
        blank = " " * width
        return [("", blank) for _ in range(height)]

# ============================================================
# Terminal helpers
# ============================================================

def get_terminal_width() -> int:
    return max(shutil.get_terminal_size(fallback=(120, 24)).columns, 40)


# ============================================================
# Persistent History (JSONL)
# ============================================================
# Design decision: JSONL over FileHistory.
# FileHistory uses a proprietary format (lines starting with +/# markers)
# and does not store timestamps, session ID, or workspace.
# JSONL (like Agy's history.jsonl) is human-readable, easily grep-able,
# and extensible for future fields. Multi-line prompts (Ctrl+J) are
# stored with \n preserved — the JSON encoding handles escaping cleanly.
#
# Schema per line (adapted from Agy history.jsonl):
# {
#   "text": "the user prompt text",
#   "timestamp": "2026-09-22T15:00:00+00:00",
#   "workspace": "/path/to/workspace"
# }

ECHO_DIR = Path.home() / ".echo"
HISTORY_FILE = ECHO_DIR / "history.jsonl"
CONFIG_FILE = ECHO_DIR / "config.json"


def load_saved_settings() -> dict:
    # Le ~/.echo/config.json. Retorna {} se nao existir ou estiver invalido.
    if not CONFIG_FILE.exists():
        return {}
    try:
        data = json.loads(CONFIG_FILE.read_text(encoding="utf-8-sig"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError) as exc:
        print(f"{YELLOW}[Warning] Could not read {CONFIG_FILE}: {exc}{RESET}")
        return {}


def save_settings(config) -> None:
    # Grava model e workspace atuais em ~/.echo/config.json.
    data = load_saved_settings()
    data["model"] = config.model
    data["workspace"] = str(config.workspace_root)
    try:
        CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_FILE.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError:
        pass

# Regex to detect paste placeholders in the buffer text.
_PASTE_PLACEHOLDER_RE = re.compile(r"\[Pasted text #(\d+) \+\d+ lines\]")


class JsonlHistory(History):
    """
    Persistent history stored in ~/.echo/history.jsonl (NDJSON format).

    Adapted from Agy (history.jsonl schema with timestamp/workspace fields).
    Rules (same as Codex chat_composer_history.rs record_local_submission_inner):
      - Empty strings are NOT saved.
      - Strings starting with space are NOT saved (shell-style secret suppression).
      - Consecutive duplicates are NOT saved (collapsed like Codex).
      - Multi-line entries (Ctrl+J) are preserved as-is via JSON escaping.
      - Slash commands ARE saved (Agy saves them; Codex does too for /model, /clear).
        Exception: /exit /quit — not useful to recall.
    """

    def __init__(self, filename: Path, workspace: Optional[str] = None):
        self.filename = filename
        self.workspace = workspace or str(Path.cwd())
        filename.parent.mkdir(parents=True, exist_ok=True)
        super().__init__()

    def load_history_strings(self):
        """Load history from JSONL file, newest-first (prompt_toolkit convention)."""
        if not self.filename.exists():
            return []

        entries = []
        try:
            with self.filename.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        text = obj.get("text", "")
                        if text:
                            entries.append(text)
                    except json.JSONDecodeError:
                        pass
        except OSError:
            pass

        # Return newest-first (prompt_toolkit expects this order).
        return list(reversed(entries))

    def store_string(self, string: str) -> None:
        """
        Append a string to the JSONL file.
        Skips: empty, leading-space, /exit /quit, consecutive duplicates.
        """
        text = string.strip()
        if not text:
            return
        if string.startswith(" "):
            return
        # Skip exit commands — not useful in history.
        if text.lower() in ("/exit", "/quit", "exit", "quit"):
            return

        # Consecutive duplicate detection: read the last line of the file.
        if self.filename.exists():
            try:
                with self.filename.open("rb") as f:
                    # Seek to find the last non-empty line efficiently.
                    try:
                        f.seek(-2, 2)
                        while f.read(1) != b"\n":
                            f.seek(-2, 1)
                        last_line = f.readline().decode("utf-8", errors="replace").strip()
                    except OSError:
                        # File is small (no newline to seek past)
                        f.seek(0)
                        last_line = f.read().decode("utf-8", errors="replace").strip()
                    if last_line:
                        try:
                            last_obj = json.loads(last_line)
                            if last_obj.get("text", "") == text:
                                return  # Consecutive duplicate — skip.
                        except json.JSONDecodeError:
                            pass
            except OSError:
                pass

        entry = {
            "text": text,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "workspace": self.workspace,
        }

        try:
            with self.filename.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError:
            pass


# ============================================================
# Slash commands
# ============================================================
# Adapted from Codex (slash_command.rs, 60+ commands) and Agy (34 slash
# commands). Echo ships a minimal-but-extensible set for Batch 2.

SLASH_COMMANDS: dict[str, str] = {
    "/help":      "Show available commands and shortcuts",
    "/clear":     "Clear the terminal screen",
    "/model":     "Switch model: /model <name>",
    "/models":    "List installed models and pick one (arrows + Enter)",
    "/workspace": "Change workspace: /workspace <path>",
    "/stats":     "Show stats from the last execution",
    "/tree":      "Show workspace directory tree",
    "/ls":        "List workspace directory contents",
    "/new":       "Start a new conversation (clear history)",
    "/exit":      "Exit Echo (alias: /quit)",
    "/quit":      "Exit Echo (alias: /exit)",
}


def _show_help() -> str:
    """Build the /help output string."""
    lines = [
        f"{BOLD}{WHITE}Echo Commands{RESET}",
        "",
    ]
    for cmd, desc in SLASH_COMMANDS.items():
        lines.append(f"  {CYAN}{cmd:<12}{RESET} {GRAY}{desc}{RESET}")
    lines += [
        "",
        f"{BOLD}{WHITE}Shortcuts{RESET}",
        "",
        f"  {CYAN}Enter      {RESET} {GRAY}Submit prompt{RESET}",
        f"  {CYAN}Ctrl+J     {RESET} {GRAY}Insert newline (multi-line){RESET}",
        f"  {CYAN}Ctrl+C     {RESET} {GRAY}Clear input / interrupt{RESET}",
        f"  {CYAN}Ctrl+D     {RESET} {GRAY}Exit{RESET}",
        f"  {CYAN}↑ / ↓      {RESET} {GRAY}Navigate history{RESET}",
        f"  {CYAN}Ctrl+R     {RESET} {GRAY}Reverse history search{RESET}",
        f"  {CYAN}/          {RESET} {GRAY}Type to see command autocomplete{RESET}",
    ]
    return "\n".join(lines)


# ============================================================
# Slash command completer (Float-based — does NOT affect layout)
# ============================================================
# CRITICAL: Uses reserve_space_for_menu=0 on the BufferControl
# so the completion dropdown is rendered as a floating overlay (Float)
# and never pushes the footer down.
# Ref: prompt_toolkit/shortcuts/prompt.py FloatContainer + CompletionsMenu
# Ref: Codex command_popup.rs (overlay modal for slash commands)

class SlashCommandCompleter(Completer):
    """
    Completer that activates only when the line starts with '/'.
    Provides fuzzy-prefix filtering for slash commands.

    Adapted from Agy SuggestionModel (fuzzy matching) and Codex
    command_popup.rs (prefix filtering on the command list).
    """

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor

        # Only activate when the entire buffer starts with '/'.
        if not text.startswith("/"):
            return

        # Extract the word being typed (first token).
        # Split on space: if there's already a space, we're in the args portion.
        parts = text.split(" ", 1)
        cmd_prefix = parts[0]

        # If they've typed a space (in argument position), don't complete.
        if len(parts) > 1:
            return

        query = cmd_prefix.lower()

        for cmd, desc in SLASH_COMMANDS.items():
            if cmd.startswith(query):
                # Show command name + description as display metadata.
                yield Completion(
                    cmd,
                    start_position=-len(cmd_prefix),
                    display=cmd,
                    display_meta=desc,
                )


# ============================================================
# Prompt UI (Custom Inline Application)
# ============================================================

PROMPT_STYLE = Style.from_dict(
    {
        "separator": "#5f5f5f",
        "footer": "#5f5f5f",
        "model": "#5f5f5f",
        # Paste placeholder styling — visually distinct from normal text.
        # Uses dim + yellow to make placeholders easy to spot.
        "paste-placeholder": "ansiyellow",
        # Completion menu styling.
        "completion-menu":                "bg:default  #d4d4d4",
        "completion-menu.completion":     "bg:default  #d4d4d4",
        "completion-menu.completion.current": "bg:default  #ffffff bold",
        "completion-menu.meta.completion": "bg:default  #808080",
        "completion-menu.meta.completion.current": "bg:default  #cccccc",
    }
)


# ============================================================
# Key bindings
# ============================================================

def create_key_bindings(echo_input: "EchoInput") -> KeyBindings:
    bindings = KeyBindings()

    @bindings.add("enter")
    def handle_enter(event):
        buffer = event.current_buffer
        text = buffer.text
        
        
        print(repr(text))

        if not text.strip():
            return

        # Texto como o usuario digitou (com placeholders de paste), para ecoar no historico.
        echo_input.last_display = text.strip()
        
        
        # Persist the submitted input so Up/Down can navigate history.
        buffer.history.append_string(text)

        # ── Slash command dispatch ──────────────────────────────────────────
        # Ref: Codex (slash_command.rs) and Agy (slash commands handler).
        # Commands are handled client-side; NOT sent to the model.
        stripped = text.strip()

        if stripped.startswith("/"):
            parts = stripped.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            if cmd == "/help":
                buffer.reset()
                event.app.exit(result="\x00/help")
                return
            elif cmd == "/clear":
                buffer.reset()
                event.app.exit(result="\x00/clear")
                return
            elif cmd in ("/exit", "/quit"):
                event.app.exit(result="\x00/exit")
                return
            elif cmd == "/model":
                if not arg:
                    buffer.reset()
                    event.app.exit(result="\x00/model?")
                    return
                buffer.reset()
                event.app.exit(result=f"\x00/model {arg}")
                return
            elif cmd == "/models":
                buffer.reset()
                event.app.exit(result="\x00/models")
                return
            elif cmd == "/workspace":
                if not arg:
                    buffer.reset()
                    event.app.exit(result="\x00/workspace?")
                    return
                buffer.reset()
                event.app.exit(result=f"\x00/workspace {arg}")
                return
            elif cmd in ("/new", "/reset"):
                buffer.reset()
                event.app.exit(result="\x00/new")
                return
            elif cmd == "/stats":
                buffer.reset()
                event.app.exit(result="\x00/stats")
                return
            elif cmd == "/tree":
                buffer.reset()
                event.app.exit(result="\x00/tree")
                return
            elif cmd == "/ls":
                buffer.reset()
                event.app.exit(result="\x00/ls")
                return
            else:
                # Unknown command — show error inline (don't exit).
                # The REPL loop will display it after exit(result=...).
                buffer.reset()
                event.app.exit(result=f"\x00/unknown {stripped}")
                return

        # ── Expand paste placeholders before submitting ────────────────────
        # Ref: Codex (chat_composer.rs handle_paste / submit expansion)
        # Ref: Agy (pending_pastes Vec<(placeholder, original)> in HistoryEntry)
        # We store placeholder→original in echo_input.pasted_texts and expand
        # them at submit time, so the model always receives the full text.
        expanded_text = _expand_paste_placeholders(text, echo_input.pasted_texts)

        event.app.exit(result=expanded_text.strip())

    @bindings.add("c-j")
    def handle_ctrl_j(event):
        """
        Multi-line input insertion via Ctrl+J.
        Reference: Agy (keybindings/defaults.go) & Codex (bottom_pane/textarea.rs).
        Allows composing multi-line inputs without submitting early.
        """
        event.current_buffer.insert_text("\n")

    @bindings.add("c-c")
    def handle_ctrl_c(event):
        """
        Ctrl+C in input buffer.
        Reference: Codex (bottom_pane/mod.rs input routing) & Agy (ActionCancel).
        - If the composer has content, Ctrl+C clears the current input buffer
          AND clears any accumulated paste references.
        - If the composer is already empty, Ctrl+C propagates KeyboardInterrupt to exit REPL.
        """
        buffer = event.current_buffer
        if buffer.text:
            buffer.reset()
            # Clear paste dict on Ctrl+C — Codex also discards paste state on cancel.
            echo_input.pasted_texts.clear()
            echo_input.paste_counter = 0
        else:
            event.app.exit(exception=KeyboardInterrupt)

    @bindings.add("c-d")
    def handle_ctrl_d(event):
        # Ctrl+D: if the buffer is empty, exit; otherwise, delete
        # the character to the right of the cursor (standard behavior).
        if not event.current_buffer.text:
            event.app.exit(exception=EOFError)
        else:
            event.current_buffer.delete()

    # ── Bracketed Paste handler ────────────────────────────────────────────
    # Ref: Codex (paste_burst.rs — detects large pastes and buffers them).
    # Ref: Agy (HistoryEntry.pending_pastes Vec<(placeholder, original)>).
    #
    # prompt_toolkit fires Keys.BracketedPaste with event.data = the full
    # pasted string when the terminal sends ESC[200~ ... ESC[201~ (bracketed
    # paste mode). This is cleaner than the Codex PasteBurst state machine
    # (which handles terminals WITHOUT bracketed paste by timing rapid chars).
    # Since WSL + Windows Terminal supports bracketed paste natively, we can
    # use this simpler approach.
    #
    # Threshold: Codex uses PASTE_BURST_MIN_CHARS=3 chars as the trigger to
    # start buffering. Agy uses a richer heuristic. We chose:
    #   lines > 5 OR chars > 300
    # This avoids replacing short pastes (URLs, filenames) with a placeholder.

    @bindings.add(Keys.BracketedPaste)
    def handle_paste(event):
        text = event.data.replace("\r\n", "\n").replace("\r", "\n")
        line_count = text.count("\n") + 1
        char_count = len(text)

        if line_count > 5 or char_count > 300:
            # Large paste → replace with placeholder.
            n = echo_input.paste_counter + 1
            echo_input.paste_counter = n
            placeholder = f"[Pasted text #{n} +{line_count} lines]"
            echo_input.pasted_texts[placeholder] = text

            # Insert only the placeholder into the buffer.
            event.current_buffer.insert_text(placeholder)
        else:
            # Small paste → insert verbatim.
            event.current_buffer.insert_text(text)

    return bindings


def _expand_paste_placeholders(text: str, pasted_texts: dict[str, str]) -> str:
    """
    Expand [Pasted text #N +K lines] placeholders back to their original text.

    Ref: Codex (chat_composer.rs: pending_pastes expansion before submit).
    Ref: Agy (HistoryEntry.pending_pastes Vec<(String, String)> — placeholder→payload).

    If the user edited the placeholder, the regex won't match and the placeholder
    text is sent as-is (accepted behavior — same as Claude Code / Codex).
    """
    if not pasted_texts:
        return text

    def replace_match(m: re.Match) -> str:
        key = m.group(0)
        return pasted_texts.get(key, key)

    return _PASTE_PLACEHOLDER_RE.sub(replace_match, text)


def create_blinking_output():
    """
    Create a native prompt_toolkit VT100 output and replace the
    default `show_cursor()` behavior.

    ─────────────────────────────────────────────────────────────
    THE BUG (source: prompt_toolkit/output/vt100.py, line ~670):

        def show_cursor(self) -> None:
            if self._cursor_visible in (False, None):
                self._cursor_visible = True
                # Stop blinking cursor and show.
                self.write_raw("\\x1b[?12l\\x1b[?25h")

    The `\\x1b[?12l` (ATT160 Reset) DISABLES the cursor blink.
    It is sent on every redraw, overriding any previous
    `\\x1b[1 q` (blinking block).

    Since `set_cursor_shape()` is only called when the SHAPE
    changes (renderer.py, lines 718-724), from the second frame
    onward only `\\x1b[?12l\\x1b[?25h` is sent — killing the blink.

    ─────────────────────────────────────────────────────────────
    THE FIX:

    We replace `show_cursor()` with a version that sends:

        \\x1b[?12h  → ATT160 Set: ENABLES cursor blink
        \\x1b[?25h  → Show cursor
        \\x1b[1 q   → DECSCUSR 1: cursor = Blinking Block

    Thus, on EVERY render, the cursor is actively reaffirmed as a
    blinking block. Works on WSL + Windows Terminal and on any
    modern xterm.
    """
    output = create_output()

    if isinstance(output, Vt100_Output):
        def _show_cursor() -> None:
            # Same condition as the original: only write if the cursor
            # is not already marked as visible.
            if output._cursor_visible in (False, None):
                output._cursor_visible = True
                # \x1b[?12h: enable blink (ATT160 Set)
                # \x1b[?25h: show cursor
                # \x1b[1 q : cursor as blinking block (DECSCUSR)
                output.write_raw("\x1b[?12h\x1b[?25h\x1b[1 q")

        output.show_cursor = _show_cursor

    return output


class EchoInput:
    """
    Create an inline 4-line layout using prompt_toolkit.
    This ensures the footer remains visible while the user types,
    without creating block bars or huge empty spaces.

    Adaptation from Agy & Codex:
    - Width calculation is dynamic (re-evaluated on every frame) to prevent
      terminal auto-wrap and cursor offset drift on terminal resize.
    - Input window uses wrap_lines=True and Dimension(min=1, max=8) to gracefully
      expand when typing long commands or multiple lines via Ctrl+J, while
      collapsing back to exactly 1 row (4 lines total) for standard prompts.
    - Persistent history via JsonlHistory (~/.echo/history.jsonl).
    - Slash command autocomplete via Float/CompletionsMenu overlay.
    - Paste placeholder support for large pastes.
    """
    def __init__(self, config: EchoConfig):
        self.config = config

        # ── Paste placeholder state ──────────────────────────────────────
        # Ref: Codex (chat_composer.rs pending_pastes, PasteBurst).
        # Ref: Agy (HistoryEntry.pending_pastes Vec<(String, String)>).
        # Dict maps placeholder string → original pasted text.
        # Cleared on Ctrl+C or after successful submit.
        self.pasted_texts: dict[str, str] = {}
        self.paste_counter: int = 0
        self.last_display: str = ""

        # ── Persistent history ────────────────────────────────────────────
        # JSONL format (not FileHistory) for richer metadata.
        # Ref: Agy ~/.gemini/antigravity-cli/history.jsonl (NDJSON schema).
        history = JsonlHistory(
            HISTORY_FILE,
            workspace=str(config.workspace_root),
        )

        # ── Buffer with history & completer ───────────────────────────────
        self.buffer = Buffer(
            history=history,
            completer=SlashCommandCompleter(),
            # complete_while_typing=True would fire on every keystroke;
            # we use False and let Tab/manual trigger drive completion.
            # This avoids layout jitter from spurious completion events.
            complete_while_typing=True,
        )

        # ── Layout construction ───────────────────────────────────────────
        # LAYOUT RULES (must not be violated):
        # 1. prompt_window height=1 (explicit, no Dimension)
        # 2. input_window Dimension(min=1, max=8) + dont_extend_height=True
        # 3. footer is always the last element of the HSplit, never floated
        # 4. CompletionsMenu is a FLOAT — never part of HSplit

        # Line 1: Top separator.
        self.top_separator = Window(
            FormattedTextControl(lambda: [("class:separator", "─" * self.get_content_width())]),
            height=1,
        )

        # Line 2: Prompt + Input (VSplit).
        self.prompt_window = Window(
            # FormattedTextControl(lambda: ANSI(f"{GREEN}➜{RESET} {WHITE}${RESET} ")),
            # width=4,  # Visible width of "➜ $ ".
            width=0,  # Set to 0 to hide the prompt; restore to 4 to show "➜ $ ".
            height=1,
            dont_extend_width=True,
        )

        # input_window: reserve_space_for_menu=0 so the CompletionsMenu
        # does NOT push the footer down — it floats instead.
        # Ref: prompt_toolkit/shortcuts/prompt.py FloatContainer pattern.
        self.input_window = Window(
            BufferControl(
                buffer=self.buffer,
                # reserve_space_for_menu=0: disable space reservation
                # for the completion menu inside the window. The menu
                # is rendered as a Float overlay instead.
            ),
            # get_line_prefix: adds 4-space indent on continuation lines.
            # Ref: Codex (bottom_pane/textarea.rs soft-wrap prefix logic).
            # Ref: Agy (editing/editing.go multi-line prompt continuation).
            # lineno=0 is the first line (has the "➜ $ " prompt_window).
            # lineno>0 are continuation lines — add 4 spaces of indent.
            get_line_prefix=lambda lineno, wrap_count: "",
            wrap_lines=True,
            height=Dimension(min=1, max=8),
            dont_extend_height=True,  # CRITICAL: prevents HSplit stretch
        )

        self.input_line = VSplit([self.prompt_window, self.input_window])

        # Line 3: Bottom separator.
        self.bottom_separator = Window(
            FormattedTextControl(lambda: [("class:separator", "─" * self.get_content_width())]),
            height=1,
        )

        # Line 4: Footer (Shortcuts + Model).
        self.footer = Window(
            FormattedTextControl(self.get_footer_text),
            height=1,
        )

        # ── Root layout (no completions menu here) ─────────────────────────
        root = HSplit([
            self.top_separator,
            self.input_line,
            # Inline completions menu — sits BETWEEN the input and the
            # bottom separator. ConditionalContainer only renders it when
            # has_completions is True, so the layout collapses back to
            # exactly 4 lines when the menu is closed.
            #
            # Ref: Agy dropdown.go (inline suggestion list attached to the
            # composer, pushing the footer down while open).
            ConditionalContainer(
                HSplit([
                    # Blank spacer line between the input and the menu.
                    # Matches Agy's dropdown.go behavior (small gap above
                    # the inline suggestion list).
                    Window(height=1),
                    Window(
                        FormattedTextControl(self._completion_fragments),
                        dont_extend_height=True,
                    ),
                ]),
                filter=has_completions,
            ),
            self.bottom_separator,
            self.footer,
        ])

        # ── CompletionsMenu as a Float overlay ──────────────────────────────
        # CompletionsMenu is itself a container (wraps its own Window), so it
        # must live inside a Float in a FloatContainer — never wrapped in a
        # plain Window, and never a direct HSplit child. This is what keeps
        # the footer pinned: the Float overlays on top without reserving
        # layout space.
        self.layout = Layout(root)

        # ── Application ───────────────────────────────────────────────────
        # FIX: Disable CPR (Cursor Position Report) to prevent the resize
        # ghost-separator bug on WSL.
        #
        # ROOT CAUSE of the duplicate-separator bug:
        # prompt_toolkit's _on_resize() calls _request_absolute_cursor_position()
        # which sends \x1b[6n (CPR request). In WSL, the CPR response either
        # arrives late or is malformed, causing the renderer to not know where
        # the current output starts. It then draws a fresh frame BELOW the
        # stale frame instead of overwriting it — duplicating the top separator.
        #
        # CODEX APPROACH (custom_terminal.rs resize + set_viewport_area):
        # Codex uses ratatui's diff_buffers() + explicit viewport tracking,
        # and never relies on CPR. On resize it calls set_viewport_area() to
        # clamp the render area and triggers a full redraw via transcript_reflow().
        #
        # PROMPT_TOOLKIT EQUIVALENT:
        # Set cpr_support = CPR_Support.NOT_SUPPORTED BEFORE running the app.
        # This makes request_absolute_cursor_position() use a fallback path
        # (relative cursor movement) that is WSL-safe. Combined with
        # renderer.reset() on resize, this eliminates the ghost frame.
        output = create_blinking_output()
        self.app = Application(
            layout=self.layout,
            key_bindings=create_key_bindings(self),
            style=PROMPT_STYLE,
            cursor=CursorShape.BLINKING_BLOCK,
            full_screen=False,
            erase_when_done=True,
            output=output,
        )

        # Disable CPR immediately after Application construction.
        # This must happen before app.run() is called.
        self.app.renderer.cpr_support = CPR_Support.NOT_SUPPORTED

        self.layout.focus(self.buffer)

    def _completion_fragments(self):
        # Menu de sugestoes desenhado a mao: coluna 0, sem padding e sem scrollbar.
        state = self.buffer.complete_state
        if not state or not state.completions:
            return []

        completions = state.completions
        current = state.complete_index
        max_rows = 8

        start = 0
        if current is not None and current >= max_rows:
            start = current - max_rows + 1
        visible = completions[start:start + max_rows]

        name_width = max(len(c.display_text) for c in visible)
        fragments = []
        for offset, c in enumerate(visible):
            selected = (start + offset) == current
            name_style = (
                "class:completion-menu.completion.current"
                if selected
                else "class:completion-menu.completion"
            )
            meta_style = (
                "class:completion-menu.meta.completion.current"
                if selected
                else "class:completion-menu.meta.completion"
            )
            fragments.append((name_style, c.display_text.ljust(name_width)))
            meta = c.display_meta_text
            if meta:
                fragments.append((meta_style, "   " + meta))
            if offset < len(visible) - 1:
                fragments.append(("", "\n"))
        return fragments

    def get_content_width(self) -> int:
        """
        Dynamically calculate content width based on current terminal size.
        Reference: Codex (custom_terminal.rs autoresize) & Agy (layout/layout.go).
        Prevents separator wrapping and line corruption on resize.
        """
        return max(get_terminal_width() - 1, 20)

    def get_footer_text(self):
        left_text = "? for shortcuts"
        right_text = self.config.model

        # The padding is calculated so that the total sum is EXACTLY
        # equal to the dynamic separator width.
        content_width = self.get_content_width()
        padding_len = max(1, content_width - len(left_text) - len(right_text))
        padding = " " * padding_len

        return [("class:footer", f"{left_text}{padding}{right_text}")]

    def prompt(self) -> str:
        """Run the application and return the typed text."""
        # Clear the buffer BEFORE running the application, so that
        # text from the previous interaction is not reused.
        self.buffer.reset()

        # Clear paste state for the new prompt cycle.
        self.pasted_texts.clear()
        self.paste_counter = 0

        result = self.app.run()

        # ----------------------------------------------------
        # HIDE the cursor when leaving the prompt.
        #
        # All streaming of the model response happens via
        # sys.stdout.write directly. If the cursor remains visible,
        # it will "follow" the text being printed.
        #
        # When prompt() is called again, the custom show_cursor()
        # (via Vt100_Output) will automatically reactivate the
        # blinking cursor.
        # ----------------------------------------------------
        sys.stdout.write(CURSOR_HIDE)
        sys.stdout.flush()

        return result or ""


# ============================================================
# Slash command execution (REPL-level)
# ============================================================

def _normalize_model_name(name: str) -> str:
    # Ollama trata "qwen2.5-coder" e "qwen2.5-coder:latest" como o mesmo modelo.
    return name if ":" in name else f"{name}:latest"


def fetch_installed_models(base_url: str, timeout: float = 5.0) -> list[dict]:
    # Lista os modelos instalados no Ollama via GET /api/tags.
    import urllib.request

    url = base_url.rstrip("/").removesuffix("/v1") + "/api/tags"
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    models = [
        {"name": m.get("name") or m.get("model", ""), "size": m.get("size", 0)}
        for m in payload.get("models", [])
    ]
    models = [m for m in models if m["name"]]
    return sorted(models, key=lambda m: m["name"].lower())


def select_model_interactive(models: list[dict], current: str) -> Optional[str]:
    # Seletor inline com setas. Retorna o nome escolhido ou None se cancelado.
    from prompt_toolkit.data_structures import Point

    if not models:
        return None

    current_norm = _normalize_model_name(current)
    index = [0]
    for i, m in enumerate(models):
        if _normalize_model_name(m["name"]) == current_norm:
            index[0] = i
            break

    def render_list():
        fragments = []
        last = len(models) - 1
        for i, m in enumerate(models):
            selected = i == index[0]
            is_current = _normalize_model_name(m["name"]) == current_norm
            marker = "> " if selected else "  "
            size = f"{m['size'] / 1e9:.1f} GB" if m.get("size") else ""
            meta = "  ".join(x for x in (size, "(current)" if is_current else "") if x)
            style = "class:picker.selected" if selected else "class:picker.item"
            fragments.append((style, f"{marker}{m['name']}"))
            fragments.append(("class:picker.meta", f"  {meta}" + ("\n" if i < last else "")))
        return fragments

    # Mesma linha em branco que fica acima do menu de completions do "/".
    spacer = Window(height=1)
    hint = Window(
        FormattedTextControl(
            [("class:picker.title", "  Up/Down to move, Enter to select, Esc to cancel")]
        ),
        height=1,
    )
    list_window = Window(
        FormattedTextControl(
            render_list,
            show_cursor=False,
            get_cursor_position=lambda: Point(0, index[0]),
        ),
        height=min(len(models), 12),
        dont_extend_height=True,
    )

    kb = KeyBindings()

    @kb.add("up")
    @kb.add("k")
    def _up(event):
        index[0] = (index[0] - 1) % len(models)
        event.app.invalidate()

    @kb.add("down")
    @kb.add("j")
    def _down(event):
        index[0] = (index[0] + 1) % len(models)
        event.app.invalidate()

    @kb.add("enter")
    def _select(event):
        event.app.exit(result=models[index[0]]["name"])

    @kb.add("escape", eager=True)
    @kb.add("c-c")
    @kb.add("q")
    def _cancel(event):
        event.app.exit(result=None)

    app = Application(
        layout=Layout(HSplit([spacer, list_window, hint]), focused_element=list_window),
        key_bindings=kb,
        style=Style.from_dict(
            {
                "picker.title": "#5f5f5f",
                "picker.item": "#d4d4d4",
                "picker.selected": "#ffffff bold",
                "picker.meta": "#808080",
            }
        ),
        full_screen=False,
        erase_when_done=True,
    )
    app.renderer.cpr_support = CPR_Support.NOT_SUPPORTED

    try:
        return app.run()
    except (KeyboardInterrupt, EOFError):
        return None


def execute_slash_command(
    cmd_result: str,
    config: EchoConfig,
    last_stats: Optional[str],
) -> tuple[bool, bool, bool]:
    """
    Execute a slash command from the REPL loop.

    Returns (should_continue, model_changed, workspace_changed):
      - should_continue: True = stay in REPL, False = exit
      - model_changed: True = config.model was updated
      - workspace_changed: True = config.workspace_root was updated
        (caller must reload the orchestrator's filesystem tools)
    """
    # Strip the internal \x00 prefix marker.
    payload = cmd_result.lstrip("\x00")

    if payload == "/help":
        print("\n" + _show_help())
        return True, False, False

    elif payload == "/clear":
        # Clear terminal screen — same as Codex /clear and Agy /clear.
        sys.stdout.write("\033[2J\033[H")
        sys.stdout.flush()
        return True, False, False

    elif payload == "/exit":
        return False, False, False

    elif payload == "/stats":
        if last_stats:
            print(f"\n{GRAY}[{last_stats}]{RESET}")
        else:
            print(f"\n{GRAY}No stats yet.{RESET}")
        return True, False, False

    elif payload == "/tree":
          from echo.tools.filesystem import FilesystemToolHandler
          handler = FilesystemToolHandler(config.workspace_root)
          print(f"\n{handler.tree('.')}")
          return True, False, False

    elif payload == "/ls":
          from echo.tools.filesystem import FilesystemToolHandler
          handler = FilesystemToolHandler(config.workspace_root)
          print(f"\n{handler.list_directory('.')}")
          return True, False, False

    elif payload == "/models":
        loading = Spinner("Loading models")
        loading_started = time.perf_counter()
        print()
        loading.start()
        try:
            models = fetch_installed_models(config.ollama_url)
            # Mantem o spinner visivel tempo suficiente para ser percebido.
            remaining = 0.4 - (time.perf_counter() - loading_started)
            if remaining > 0:
                time.sleep(remaining)
        except (OSError, ValueError) as exc:
            loading.stop()
            sys.stdout.write("\033[1A")
            sys.stdout.flush()
            print(f"\n{RED}Could not list models from {config.ollama_url}: {exc}{RESET}")
            return True, False, False
        finally:
            loading.stop()
        sys.stdout.write("\033[1A")
        sys.stdout.flush()

        if not models:
            print(f"\n{YELLOW}No models installed. Run: ollama pull <name>{RESET}")
            return True, False, False

        chosen = select_model_interactive(models, config.model)
        if chosen and _normalize_model_name(chosen) != _normalize_model_name(config.model):
            config.model = chosen  # type: ignore[attr-defined]
            save_settings(config)
            print(f"\n{GREEN}Model switched to: {chosen}{RESET}")
            return True, True, False

        print(f"\n{GRAY}Model unchanged: {config.model}{RESET}")
        return True, False, False

    elif payload == "/model?":
        print(f"\n{YELLOW}Usage: /model <name>{RESET}")
        print(f"{GRAY}Current model: {config.model}{RESET}")
        return True, False, False

    elif payload.startswith("/model "):
        new_model = payload[len("/model "):].strip()
        if new_model:
            config.model = new_model  # type: ignore[attr-defined]
            save_settings(config)
            print(f"\n{GREEN}Model switched to: {new_model}{RESET}")
            return True, True, False
        return True, False, False

    elif payload == "/workspace?":
        print(f"\n{YELLOW}Usage: /workspace <path>{RESET}")
        print(f"{GRAY}Current workspace: {config.workspace_root}{RESET}")
        return True, False, False

    elif payload.startswith("/workspace "):
        new_ws = payload[len("/workspace "):].strip()
        new_path = Path(new_ws).expanduser().resolve()
        if new_path.exists() and new_path.is_dir():
            config.workspace_root = new_path  # type: ignore[attr-defined]
            save_settings(config)
            print(f"\n{GREEN}Workspace changed to: {new_path}{RESET}")
            return True, False, True
        else:
            print(f"\n{RED}Error: '{new_ws}' is not a valid directory.{RESET}")
            return True, False, False

    elif payload.startswith("/unknown "):
        unknown_cmd = payload[len("/unknown "):].strip()
        # Extract just the command part.
        cmd_name = unknown_cmd.split()[0] if unknown_cmd else unknown_cmd
        print(f"\n{RED}Unknown command: {cmd_name}{RESET}")
        print(f"{GRAY}Type /help to see available commands.{RESET}")
        return True, False, False

    return True, False, False


# ============================================================
# Statistics
# ============================================================

def format_stats(result) -> str:
    tool_count = len(result.tool_executions)
    tool_duration = sum(execution.duration_seconds for execution in result.tool_executions)
    total_duration = result.total_duration_seconds
    response_chars = len(result.final_response or "")
    iterations = result.iterations

    iteration_label = "iteration" if iterations == 1 else "iterations"
    tool_label = "tool" if tool_count == 1 else "tools"

    return (
        f"{iterations} {iteration_label} | "
        f"{tool_count} {tool_label} | "
        f"tool time {tool_duration:.2f}s | "
        f"response {response_chars} chars | "
        f"total {total_duration:.2f}s | "
        f"{result.stopped_reason}"
    )


# ============================================================
# Banner
# ============================================================

def print_banner(config: Optional[EchoConfig] = None):
    info_lines = [
        f"{BOLD}{WHITE}Echo CLI 0.1.0{RESET}",
        f"{GRAY}Local AI Support Assistant{RESET}",
    ]

    if config is not None:
        info_lines.append(f"{GRAY}{config.model} (via Ollama @ {config.ollama_url}){RESET}")
        info_lines.append(f"{GRAY}Workspace: {config.workspace_root}{RESET}")
    else:
        info_lines.append(f"{GRAY}Qwen 2.5 3B (via Ollama){RESET}")
        info_lines.append(f"{GRAY}workspace: ./workspace{RESET}")

    logo_height = len(ECHO_LOGO)
    padded_info = info_lines + [""] * max(logo_height - len(info_lines), 0)

    print()
    for logo_line, info_line in zip(ECHO_LOGO, padded_info):
        print(f"  {WHITE}{logo_line}{RESET}   {info_line}")
    print()


# ============================================================
# Streaming UI
# ============================================================

class StreamingUI:
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.spinner = Spinner("Loading")
        self._printing = False
        self._after_tool = False  # a tool call was just shown
        self._gap = False         # blank line already printed before the spinner

    def reset(self):
        """
        Reset streaming UI state and ensure background spinner thread stops.
        Reference: Codex (chatwidget.rs update_task_running_state).
        """
        self.spinner.stop()
        self._printing = False
        self._after_tool = False
        self._gap = False

    def confirm_tool(self, tool_name: str, arguments: dict) -> bool:
        """Ask the user for confirmation before executing a sensitive tool."""
        self.spinner.stop()

        if self._printing:
            sys.stdout.write("\n")
            sys.stdout.flush()
            self._printing = False

        print()

        if tool_name == "write_file":
            path = arguments.get("path", "?")
            content = arguments.get("content", "")
            size = len(content) if isinstance(content, str) else 0

            print(f"{YELLOW}[Confirm] write_file{RESET}")
            print(f"{GRAY}  Path: {path}{RESET}")
            print(f"{GRAY}  Content: {size} characters{RESET}")

        elif tool_name == "edit_file":
            path = arguments.get("path", "?")
            old_text = arguments.get("old_text", "")
            new_text = arguments.get("new_text", "")

            print(f"{YELLOW}[Confirm] edit_file{RESET}")
            print(f"{GRAY}  Path: {path}{RESET}")
            print(
                f"{GRAY}  Change: {len(old_text)} -> "
                f"{len(new_text)} characters{RESET}"
            )

        else:
            print(f"{YELLOW}[Confirm] {tool_name}{RESET}")
            print(f"{GRAY}  Arguments: {arguments}{RESET}")

        try:
            answer = prompt(
                "  Proceed? [y/N] ",
                default="",
            )
        except (KeyboardInterrupt, EOFError):
            print()
            return False
        
        return answer.strip().lower() in ("y", "yes")
    
    

    def _lead(self) -> str:
        # Quebra de linha inicial, omitida se o gap apos a tool ja foi impresso.
        if self._gap:
            self._gap = False
            return ""
        return "\n"

    def handle_event(self, event_type: str, data: dict):
        if event_type == "run_started":
            self.reset()
        elif event_type == "iteration_started":
            if self._after_tool:
                print()
                self._after_tool = False
            # A linha do spinner funciona como a linha em branco de separacao:
            # o proximo texto comeca nela, sem \n extra.
            self._gap = True
            self.spinner.start("Loading")
        elif event_type == "content_delta":
            self.spinner.stop()
            if not self._printing:
                sys.stdout.write(self._lead())
                self._printing = True
            sys.stdout.write(f"{WHITE}{data['delta']}{RESET}")
            sys.stdout.flush()

        elif event_type == "tool_call_received":
            self.spinner.stop()
            lead = self._lead()

            if self.verbose:
                self._after_tool = True

                tool_name = data["tool_name"]
                arguments = data["arguments"]

                try:
                    if isinstance(arguments, str):
                        arguments = json.loads(arguments)

                    formatted_arguments = json.dumps(
                        arguments,
                        ensure_ascii=False,
                        indent=2,
                    )
                except (TypeError, ValueError, json.JSONDecodeError):
                    formatted_arguments = str(arguments)



                print(f"{lead}{GRAY}[Tool Call]{RESET}")
                print(f"{GRAY}{tool_name}({formatted_arguments}){RESET}")
        
        elif event_type == "tool_call_executed":
            if self.verbose:
                status = "OK" if data["success"] else "FAILED"
                result_text = str(data["result"])
                is_tree = data.get("tool_name") == "tree"
                if not is_tree:
                    result_lines = result_text.splitlines()
                    if len(result_lines) > 15:
                        hidden = len(result_lines) - 15
                        result_text = "\n".join(result_lines[:15]) + f"\n... (+{hidden} lines hidden)"
                        
                        
                header = f"{GRAY}[Tool Executed ({status}) in {data['duration']:.3f}s]{RESET}"

                if is_tree:
                    print(f"\n{header}\n{WHITE}{result_text}{RESET}")
                else:
                    print(f"\n{header} {GRAY}-> {result_text}{RESET}")
                    
                    
        elif event_type == "workspace_reloaded":
            print(f"\n{GRAY}[Tools reloaded -> {data['workspace_root']}]{RESET}")
        elif event_type == "error":
            self.spinner.stop()
            print(f"\n{self._lead()}{BOLD}{RED}[Error]{RESET} {data.get('error')}")
        elif event_type in ("run_completed", "max_iterations_reached"):
            self.spinner.stop()
            if self._printing:
                sys.stdout.write("\n")
                sys.stdout.flush()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Echo — Local AI Support Assistant with Tool Execution")
    parser.add_argument("prompt", nargs="?", default=None, help="User prompt to execute. If omitted, runs in interactive mode.")
    parser.add_argument("--workspace", "-w", default="./workspace", help="Workspace directory path")
    parser.add_argument("--model", "-m", default="qwen2.5-coder", help="Ollama model name")
    parser.add_argument("--url", "-u", default="http://localhost:11434", help="Ollama server URL")
    parser.add_argument("--verbose", "-v", action="store_true", default=True, help="Print verbose execution tracing")
    parser.add_argument("--quiet", "-q", action="store_true", help="Disable verbose output")
    parser.add_argument("--check-health", action="store_true", help="Check Ollama connection and model availability, then exit")

    args = parser.parse_args()
    verbose = args.verbose and not args.quiet

    def _passed(*flags):
        for a in sys.argv[1:]:
            for f in flags:
                if a == f or (f.startswith("--") and a.startswith(f + "=")):
                    return True
        return False

    saved = load_saved_settings()

    model = args.model
    if not _passed("--model", "-m") and saved.get("model"):
        model = saved["model"]

    workspace = args.workspace
    if not _passed("--workspace", "-w"):
        saved_ws = saved.get("workspace")
        if saved_ws and Path(saved_ws).is_dir():
            workspace = saved_ws

    config = EchoConfig(
        ollama_url=args.url,
        model=model,
        workspace_root=Path(workspace).resolve(),
    )

    client = OllamaClient(base_url=config.ollama_url)

    if args.check_health:
        print(f"Checking Ollama connection at {config.ollama_url}...")
        ok, msg = client.check_health(config.model)
        if ok:
            print(f"[HEALTH OK] {msg}")
            sys.exit(0)
        print(f"[HEALTH ERROR] {msg}")
        sys.exit(1)

    ui = StreamingUI(verbose=verbose)
    orchestrator = EchoOrchestrator(
    config=config,
    client=client,
    on_event=ui.handle_event,
    confirm_tool=ui.confirm_tool,
)
    orchestrator.tool_output_visible = verbose

    # Non-interactive mode (one-shot).
    if args.prompt:
        if verbose:
            print(f"Workspace: {config.workspace_root}")
            print(f"Model:     {config.model}")
            print(f"Prompt:    {args.prompt}")
            print()

        try:
            result = orchestrator.run(args.prompt)
            print()

            if verbose:
                print(f"{GRAY}[{format_stats(result)}]{RESET}")

            sys.exit(0 if result.stopped_reason == "completed" else 1)
        except KeyboardInterrupt:
            ui.reset()
            print(f"\n{GRAY}[Interrupted]{RESET}")
            sys.exit(130)

    # Interactive mode (REPL).
    print_banner(config)
    echo_input = EchoInput(config)
    last_stats: Optional[str] = None

    try:
        while True:
            try:
                # Ensure the cursor is on a completely new line
                # before drawing the prompt block.
                print()

                user_input = echo_input.prompt()

                if not user_input.strip():
                    continue

                # O frame do prompt foi apagado ao enviar; ecoa a entrada no historico.
                shown = echo_input.last_display or user_input
                echo_lines = shown.splitlines() or [shown]
                print(f"{GRAY}> {echo_lines[0]}{RESET}")
                for extra in echo_lines[1:]:
                    print(f"{GRAY}  {extra}{RESET}")
                echo_input.last_display = ""

                # ── Slash command handling ────────────────────────────────
                # Commands arrive with a \x00 prefix (internal marker set
                # in handle_enter bindings). The REPL executes them without
                # sending to the model.
                if user_input == "\x00/new":
                    orchestrator.reset_history()
                    print(f"\n{GREEN}Conversation cleared.{RESET}")
                    continue

                if user_input.startswith("\x00"):
                    should_continue, _model_changed, workspace_changed = (
                        execute_slash_command(
                            user_input, config, last_stats
                        )
                    )
                    if not should_continue:
                        print("Goodbye!")
                        break
                    if workspace_changed:
                        # Re-register filesystem tools so they operate on
                        # the new directory. Without this, list_directory,
                        # read_file, etc. keep using the old workspace_root.
                        # Ref: Codex (app_server_session.rs cwd change
                        # re-initialization) and Agy (store.go reload).
                        orchestrator.reload_workspace()
                    continue

                # ── Legacy exit keywords ──────────────────────────────────
                if user_input.lower() in ("exit", "quit"):
                    print("Goodbye!")
                    break

                # Blank line before the model output.
                print()

                try:
                    result = orchestrator.run(user_input)
                    print()

                    if verbose:
                        last_stats = format_stats(result)
                        print(f"{GRAY}[{last_stats}]{RESET}")
                except KeyboardInterrupt:
                    # Reference: Codex (chatwidget.rs interrupt handling) & Agy (ActionCancel).
                    # Ctrl+C during model generation/streaming cancels the active turn
                    # cleanly without terminating the Echo CLI session.
                    ui.reset()
                    print(f"\n{GRAY}[Interrupted]{RESET}")
                    continue

            except KeyboardInterrupt:
                # Ctrl+C on an empty prompt exits the REPL session cleanly.
                print("\nExiting...")
                break
            except EOFError:
                print("\nExiting...")
                break
    finally:
        # Restore the cursor to visible + default user shape when
        # leaving the REPL. Without this, the terminal may be left
        # with an invisible cursor or a "blinking block" shape.
        sys.stdout.write(CURSOR_RESTORE)
        sys.stdout.flush()


if __name__ == "__main__":
    main()
