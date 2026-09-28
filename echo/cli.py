"""Command-line interface for Echo."""

import argparse
import json
import re
import shutil
import random
import math
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.filters import has_completions, Condition
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
    ScrollablePane,
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
from .logo_frames import ECHO_LOGO, ECHO_LOGO_FRAMES


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
        self._status_sink = None

    def set_status_sink(self, callback) -> None:
        self._status_sink = callback

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

            if self._status_sink is not None:
                self._status_sink(f"{spinner_char} {loading}")
            else:
                sys.stdout.write(f"\r{GRAY}{spinner_char}{RESET} {loading}")
                sys.stdout.flush()


            self._frame += 1
            time.sleep(0.14)

        if self._status_sink is not None:
            self._status_sink("")
        else:
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


class LiveTranscript:
    """Thread-safe model output rendered inside the persistent prompt UI."""

    def __init__(self, max_lines: int = 0):
        # 0 = keep the entire conversation; the viewport scrolls instead.
        self.max_lines = max_lines
        self._text = ""
        self._status = ""
        self._lock = threading.Lock()
        self._invalidate = None

    def set_invalidator(self, callback) -> None:
        self._invalidate = callback

    def append(self, text: str) -> None:
        if not text:
            return
        with self._lock:
            self._text += text
        self._refresh()

    def set_status(self, text: str) -> None:
        with self._lock:
            self._status = text
        self._refresh()

    def clear(self) -> None:
        with self._lock:
            self._text = ""
            self._status = ""
        self._refresh()

    def has_content(self) -> bool:
        with self._lock:
            return bool(self._text or self._status)

    def _refresh(self) -> None:
        callback = self._invalidate
        if callback is not None:
            try:
                callback()
            except Exception:
                pass

    def fragments(self):
        with self._lock:
            text = self._text
            status = self._status

        # Normalize trailing newlines so the transcript itself does not
        # create arbitrary empty rows. When a status (Loading) is active,
        # explicitly reserve exactly one blank line before it.
        if status:
            base = text.rstrip("\n")
            lines = base.split("\n") if base else []
            if lines:
                lines.append("")
            lines.append(f"{GRAY}{status}{RESET}")
        else:
            lines = text.rstrip("\n").split("\n") if text else []

        if self.max_lines > 0:
            lines = lines[-self.max_lines:]

        return ANSI("\n".join(lines))


class TranscriptScrollablePane(ScrollablePane):
    """Scrollable conversation viewport with a visible scrollbar."""

    def __init__(self, content, **kwargs):
        super().__init__(
            content,
            keep_cursor_visible=False,
            keep_focused_window_visible=False,
            show_scrollbar=True,
            display_arrows=False,
            **kwargs,
        )
        # Hide the scrollbar until the transcript actually overflows.
        self.show_scrollbar = lambda: self._has_overflow

        self._follow_end = True
        self._virtual_height = 0
        self._viewport_height = 0
        self._has_overflow = False

    def write_to_screen(self, screen, mouse_handlers, write_position,
                        parent_style, erase_bg, z_index):
        # First measure using the full width. If the content overflows, enable
        # the scrollbar and remeasure with its one-column reservation.
        full_width = write_position.width
        virtual_height = self.content.preferred_height(
            full_width, self.max_available_height
        ).preferred
        virtual_height = max(virtual_height, write_position.height)
        self._has_overflow = virtual_height > write_position.height

        virtual_width = (
            write_position.width - 1
            if self._has_overflow
            else write_position.width
        )
        virtual_height = self.content.preferred_height(
            virtual_width, self.max_available_height
        ).preferred
        virtual_height = max(virtual_height, write_position.height)
        virtual_height = min(virtual_height, self.max_available_height)

        self._virtual_height = virtual_height
        self._viewport_height = write_position.height
        max_scroll = max(0, virtual_height - write_position.height)

        if self._follow_end:
            self.vertical_scroll = max_scroll
        else:
            self.vertical_scroll = max(
                0, min(self.vertical_scroll, max_scroll)
            )

        super().write_to_screen(
            screen,
            mouse_handlers,
            write_position,
            parent_style,
            erase_bg,
            z_index,
        )

    def scroll_by(self, delta: int) -> None:
        """Scroll by visual rows and stop following the live tail."""
        max_scroll = max(
            0, self._virtual_height - self._viewport_height
        )
        if max_scroll == 0:
            self._follow_end = True
            self.vertical_scroll = 0
            return

        self._follow_end = False
        self.vertical_scroll = max(
            0,
            min(self.vertical_scroll + delta, max_scroll),
        )

        if self.vertical_scroll >= max_scroll:
            self._follow_end = True

    def scroll_to_end(self) -> None:
        self._follow_end = True
        self.vertical_scroll = max(
            0, self._virtual_height - self._viewport_height
        )


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

        # Conversation scrollbar: dark track + light thumb.
        # The margin is rendered with spaces, so the colors must be applied
        # as terminal backgrounds rather than only as foreground colors.
        "scrollbar.background": "bg:#303030 #303030",
        "scrollbar.button": "bg:#909090 #909090",
        "scrollbar.arrow": "#707070",
    }
)


# ============================================================
# Key bindings
# ============================================================

def create_key_bindings(echo_input: "EchoInput") -> KeyBindings:
    bindings = KeyBindings()

    # Transcript scrolling. The viewport owns the scroll state, while
    # the input remains focused so typing is never interrupted.
    @bindings.add(Keys.ScrollUp, eager=True, is_global=True)
    def handle_scroll_up(event):
        # Mouse wheel: move only a few visual rows at a time.
        echo_input.transcript_scroll.scroll_by(-2)
        event.app.invalidate()

    @bindings.add(Keys.ScrollDown, eager=True, is_global=True)
    def handle_scroll_down(event):
        echo_input.transcript_scroll.scroll_by(2)
        event.app.invalidate()

    @bindings.add(Keys.PageUp, eager=True, is_global=True)
    def handle_page_up(event):
        echo_input.transcript_scroll.scroll_by(-4)
        event.app.invalidate()

    @bindings.add(Keys.PageDown, eager=True, is_global=True)
    def handle_page_down(event):
        echo_input.transcript_scroll.scroll_by(4)
        event.app.invalidate()

    @bindings.add(Keys.ControlHome, eager=True)
    def handle_transcript_home(event):
        echo_input.transcript_scroll._follow_end = False
        echo_input.transcript_scroll.vertical_scroll = 0
        event.app.invalidate()

    @bindings.add(Keys.ControlEnd, eager=True)
    def handle_transcript_end(event):
        echo_input.transcript_scroll.scroll_to_end()
        event.app.invalidate()

    @bindings.add("y", filter=Condition(lambda: echo_input.confirmation_active), eager=True)
    @bindings.add("Y", filter=Condition(lambda: echo_input.confirmation_active), eager=True)
    def handle_confirm_yes(event):
        echo_input.set_confirmation_choice(True)

    @bindings.add("n", filter=Condition(lambda: echo_input.confirmation_active), eager=True)
    @bindings.add("N", filter=Condition(lambda: echo_input.confirmation_active), eager=True)
    def handle_confirm_no(event):
        echo_input.set_confirmation_choice(False)

    @bindings.add("enter", filter=Condition(lambda: echo_input.confirmation_active), eager=True)
    def handle_confirm_enter(event):
        choice = echo_input.confirmation_buffer.text.strip().lower()
        if choice == "y":
            echo_input.resolve_confirmation(True)
        elif choice == "n":
            echo_input.resolve_confirmation(False)
        else:
            # Empty input follows apt-style [Y/n]: Enter means yes.
            echo_input.resolve_confirmation(echo_input._confirmation_choice)

    @bindings.add("escape", filter=Condition(lambda: echo_input.confirmation_active), eager=True)
    def handle_confirm_escape(event):
        echo_input.resolve_confirmation(False)

    @bindings.add("enter")
    def handle_enter(event):
        buffer = event.current_buffer
        text = buffer.text
        
        
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
                echo_input.submit_text(event, "\x00/help")
                return
            elif cmd == "/clear":
                buffer.reset()
                echo_input.submit_text(event, "\x00/clear")
                return
            elif cmd in ("/exit", "/quit"):
                echo_input.submit_text(event, "\x00/exit")
                return
            elif cmd == "/model":
                if not arg:
                    buffer.reset()
                    echo_input.submit_text(event, "\x00/model?")
                    return
                buffer.reset()
                echo_input.submit_text(event, f"\x00/model {arg}")
                return
            elif cmd == "/models":
                buffer.reset()
                echo_input.submit_text(event, "\x00/models")
                return
            elif cmd == "/workspace":
                if not arg:
                    buffer.reset()
                    echo_input.submit_text(event, "\x00/workspace?")
                    return
                buffer.reset()
                echo_input.submit_text(event, f"\x00/workspace {arg}")
                return
            elif cmd in ("/new", "/reset"):
                buffer.reset()
                echo_input.submit_text(event, "\x00/new")
                return
            elif cmd == "/stats":
                buffer.reset()
                echo_input.submit_text(event, "\x00/stats")
                return
            elif cmd == "/tree":
                buffer.reset()
                echo_input.submit_text(event, "\x00/tree")
                return
            elif cmd == "/ls":
                buffer.reset()
                echo_input.submit_text(event, "\x00/ls")
                return
            else:
                # Unknown command — show error inline (don't exit).
                # The REPL loop will display it after exit(result=...).
                buffer.reset()
                echo_input.submit_text(event, f"\x00/unknown {stripped}")
                return

        # ── Expand paste placeholders before submitting ────────────────────
        # Ref: Codex (chat_composer.rs handle_paste / submit expansion)
        # Ref: Agy (pending_pastes Vec<(placeholder, original)> in HistoryEntry)
        # We store placeholder→original in echo_input.pasted_texts and expand
        # them at submit time, so the model always receives the full text.
        expanded_text = _expand_paste_placeholders(text, echo_input.pasted_texts)

        echo_input.submit_text(event, expanded_text.strip())

    @bindings.add("?", filter=Condition(lambda: not echo_input.buffer.text), eager=True)
    def handle_shortcuts(event):
        """Toggle the shortcuts panel without touching conversation history."""
        echo_input.show_shortcuts = not echo_input.show_shortcuts
        event.app.invalidate()

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
            # Preserve the current terminal contents when leaving on Ctrl+C.
            event.app.erase_when_done = False
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
            # Bracketed-paste can leave a selection active in prompt_toolkit
            # (especially with large pastes). The placeholder must behave like
            # ordinary typed text, never like a selected region.
            event.current_buffer.exit_selection()
        else:
            # Small paste → insert verbatim.
            event.current_buffer.insert_text(text)
            event.current_buffer.exit_selection()

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


def create_cursorless_output():
    """Keep the physical terminal cursor hidden during prompt_toolkit redraws."""
    output = create_output()

    if isinstance(output, Vt100_Output):
        # The software cursor is rendered by BlinkingCursorBufferControl.
        # The physical cursor must stay hidden, otherwise every logo
        # animation invalidate() can interfere with its native blink.
        output.write_raw("\033[?25l")
        output._cursor_visible = False

        def _hide_cursor() -> None:
            output._cursor_visible = False

        def _show_cursor() -> None:
            output._cursor_visible = False

        output.hide_cursor = _hide_cursor
        output.show_cursor = _show_cursor

    return output


class FakeCursorBlink:
    """Software blinking cursor independent of terminal cursor state."""

    def __init__(self, interval: float = 0.53, typing_hold: float = 0.55):
        self.interval = interval
        self.typing_hold = typing_hold
        self.visible = True
        self._typing_until = 0.0
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._app: Optional[Application] = None
        self._lock = threading.Lock()

    def start(self, app: Application) -> None:
        self.stop()
        with self._lock:
            self._app = app
            self.visible = True
            self._typing_until = 0.0
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run,
                daemon=True,
                name="echo-fake-cursor",
            )
            self._thread.start()
        app.invalidate()

    def mark_typing(self) -> None:
        """Keep the fake cursor solid while text is actively being edited."""
        with self._lock:
            self.visible = True
            self._typing_until = time.perf_counter() + self.typing_hold
            app = self._app

        if app is not None:
            try:
                app.invalidate()
            except Exception:
                pass

    def resume_blinking(self) -> None:
        with self._lock:
            self._typing_until = 0.0
            self.visible = True
            app = self._app

        if app is not None:
            try:
                app.invalidate()
            except Exception:
                pass

    def _run(self) -> None:
        next_toggle = time.perf_counter() + self.interval

        while not self._stop_event.wait(0.04):
            now = time.perf_counter()

            with self._lock:
                typing = now < self._typing_until

                if typing:
                    # While typing, the cursor is always solid and the normal
                    # blink schedule is restarted from the end of the typing
                    # hold window.
                    self.visible = True
                    next_toggle = now + self.interval
                elif now >= next_toggle:
                    self.visible = not self.visible
                    next_toggle = now + self.interval

                app = self._app

            if app is not None:
                try:
                    app.invalidate()
                except Exception:
                    pass

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=1.0)

        with self._lock:
            self._thread = None
            self._app = None
            self.visible = True
            self._typing_until = 0.0


class BlinkingCursorBufferControl(BufferControl):
    """Render a software cursor at the real buffer cursor position."""

    def __init__(self, *args, cursor_blink: FakeCursorBlink, **kwargs):
        super().__init__(*args, **kwargs)
        self._cursor_blink = cursor_blink

    def create_content(self, width: int, height: int, preview_search: bool = False):
        content = super().create_content(width, height, preview_search)
        cursor = content.cursor_position
        original_get_line = content.get_line

        def get_line(lineno: int):
            fragments = list(original_get_line(lineno))

            if lineno != cursor.y or not self._cursor_blink.visible:
                return fragments

            chars: list[tuple[str, str]] = []
            for style, text in fragments:
                for char in text:
                    chars.append((style, char))

            # The cursor can sit one position beyond the last character.
            # Extend the rendered line so the software cursor is visible
            # even when the input buffer is empty.
            if cursor.x >= len(chars):
                chars.extend(
                    [("", " ")] * (cursor.x - len(chars) + 1)
                )

            chars[cursor.x] = ("class:fake-cursor", "█")
            return chars

        content.get_line = get_line
        content.show_cursor = False
        return content

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
    def __init__(self, config: EchoConfig, animate_logo: bool = True):
        self.config = config
        self._cursor_blink = FakeCursorBlink()
        self.logo_animator = LogoAnimator(
            config,
            enabled=animate_logo and sys.stdout.isatty(),
        )

        # ── Paste placeholder state ──────────────────────────────────────
        # Ref: Codex (chat_composer.rs pending_pastes, PasteBurst).
        # Ref: Agy (HistoryEntry.pending_pastes Vec<(String, String)>).
        # Dict maps placeholder string → original pasted text.
        # Cleared on Ctrl+C or after successful submit.
        self.pasted_texts: dict[str, str] = {}
        self.paste_counter: int = 0
        self.last_display: str = ""
        self.on_submit = None
        self.show_shortcuts = False
        self.transcript = LiveTranscript(max_lines=0)
        self._submit_thread: Optional[threading.Thread] = None

        # Tool confirmation is resolved inside the main prompt_toolkit app.
        # The model worker blocks on this event instead of opening a second
        # prompt() and fighting the live renderer/cursor.
        self._confirmation_event = threading.Event()
        self._confirmation_result = False
        self._confirmation_active = False
        self._confirmation_message = ""
        self._confirmation_choice = True

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
        self.buffer.on_text_changed += lambda _buffer: self._cursor_blink.mark_typing()
        self.buffer.on_cursor_position_changed += lambda _buffer: self._cursor_blink.mark_typing()

        # ── Layout construction ───────────────────────────────────────────
        # LAYOUT RULES (must not be violated):
        # 1. prompt_window height=1 (explicit, no Dimension)
        # 2. input_window Dimension(min=1, max=8) + dont_extend_height=True
        # 3. footer is always the last element of the HSplit, never floated
        # 4. CompletionsMenu is a FLOAT — never part of HSplit

        # Animated banner. The logo is part of the prompt_toolkit layout,
        # so frame updates do not write directly to stdout and cannot corrupt
        # the prompt.
        self.logo_window = Window(
            FormattedTextControl(self._logo_fragments),
            height=len(ECHO_LOGO),
            dont_extend_height=True,
        )

        # Flexible conversation viewport. It is intentionally ALWAYS
        # present, even when empty. This keeps the input/footer anchored at
        # the bottom of the terminal from the very first render.
        self.transcript_window = Window(
            FormattedTextControl(
                self.transcript.fragments,
                show_cursor=False,
            ),
            # Keep one empty column between the response text and scrollbar.
            right_margins=[SpacesMargin(1)],
            # Let ScrollablePane determine the wrapped content height.
            wrap_lines=True,
            dont_extend_height=False,
        )

        self.transcript_scroll = TranscriptScrollablePane(
            self.transcript_window,
            height=Dimension(min=0, weight=1),
        )

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
            BlinkingCursorBufferControl(
                buffer=self.buffer,
                cursor_blink=self._cursor_blink,
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

        self.confirmation_buffer = Buffer(
            read_only=False,
            multiline=False,
            max_number_of_completions=0,
        )
        confirmation_text = "Do you want to continue? [Y/n] "
        self.confirmation_input_line = VSplit([
            Window(
                FormattedTextControl(lambda: ANSI(confirmation_text)),
                width=len(confirmation_text),
                height=1,
                dont_extend_width=True,
            ),
            Window(
                BlinkingCursorBufferControl(
                    buffer=self.confirmation_buffer,
                    cursor_blink=self._cursor_blink,
                ),
                height=1,
                dont_extend_height=True,
            ),
        ])

        # Fixed visual gap between conversation output and the prompt cursor.
        # This prevents the fake cursor from appearing attached to the last
        # response line or to the execution statistics.
        self.prompt_gap = ConditionalContainer(
            Window(height=1),
            filter=Condition(lambda: not self.confirmation_active),
        )

        # Tool confirmation is a temporary bottom-docked panel. The details
        # stay immediately above the confirmation input line, rather than
        # becoming part of the scrollable conversation transcript.
        self.confirmation_details_window = ConditionalContainer(
            Window(
                FormattedTextControl(
                    lambda: ANSI(self._confirmation_message)
                ),
                height=Dimension(min=1, max=8),
                dont_extend_height=True,
            ),
            filter=Condition(lambda: self.confirmation_active),
        )

        self.normal_input_container = ConditionalContainer(
            self.input_line,
            filter=Condition(lambda: not self.confirmation_active),
        )

        # Toggleable shortcuts panel. It is separate from the transcript, so
        # pressing '?' repeatedly never adds anything to conversation history.
        self.shortcuts_window = ConditionalContainer(
            Window(
                FormattedTextControl(lambda: ANSI(_show_help())),
                wrap_lines=False,
                dont_extend_height=True,
            ),
            filter=Condition(lambda: self.show_shortcuts),
        )

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

        # ── Root layout ───────────────────────────────────────────────────
        # The input/footer are anchored at the bottom. Therefore the
        # completion dropdown must be ABOVE the input, never below it.
        root = HSplit([
            self.logo_window,
            self.top_separator,

            # '?' is a top-level help panel: it belongs immediately below
            # the banner and above the conversation/history area.
            self.shortcuts_window,

            # Visual separation between the shortcuts panel and conversation.
            Window(height=1),

            self.transcript_scroll,

            # '/' autocomplete is a prompt-only panel: it belongs immediately
            # above the bottom prompt and never shares the help panel's space.
            ConditionalContainer(
                HSplit([
                    Window(
                        FormattedTextControl(self._completion_fragments),
                        dont_extend_height=True,
                    ),
                    Window(height=1),
                ]),
                filter=has_completions,
            ),

            self.confirmation_details_window,
            self.prompt_gap,
            ConditionalContainer(
                self.confirmation_input_line,
                filter=Condition(lambda: self.confirmation_active),
            ),
            self.normal_input_container,
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
        output = create_cursorless_output()
        self.app = Application(
            layout=self.layout,
            key_bindings=create_key_bindings(self),
            style=PROMPT_STYLE,
            cursor=CursorShape._NEVER_CHANGE,
            full_screen=False,
            erase_when_done=False,
            # Keep terminal-native mouse selection/copy. The scrollbar remains
            # visible and keyboard scrolling remains available, but the
            # application no longer captures drag/click events from the
            # terminal emulator.
            mouse_support=False,
            output=output,
        )

        # Disable CPR immediately after Application construction.
        # This must happen before app.run() is called.
        self.app.renderer.cpr_support = CPR_Support.NOT_SUPPORTED
        self.transcript.set_invalidator(self.app.invalidate)

        self.layout.focus(self.buffer)

        # Start the logo animation immediately. The prompt_toolkit application
        # renders the prompt and animated banner together, so there is no
        # startup animation delay before the prompt becomes usable.
        self.start_logo_animation()

    def _logo_fragments(self):
        # FormattedTextControl accepts ANSI-formatted text directly.
        return ANSI("\n".join(self.logo_animator.current_lines()))

    def start_logo_animation(self):
        self.logo_animator.start(self.app)

    def stop_logo_animation(self):
        self.logo_animator.stop()

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

    @property
    def confirmation_active(self) -> bool:
        return self._confirmation_active

    def request_confirmation(self, message: str) -> bool:
        """Show confirmation details above the input and wait for its answer."""
        self._confirmation_result = False
        self._confirmation_choice = True
        self._confirmation_event.clear()
        self._confirmation_message = message
        self._confirmation_active = True
        self.confirmation_buffer.reset()
        self.app.layout.focus(self.confirmation_buffer)
        self._cursor_blink.mark_typing()
        self.app.invalidate()

        self._confirmation_event.wait()
        result = self._confirmation_result

        self._confirmation_active = False
        self._confirmation_message = ""
        self._cursor_blink.visible = True
        self.app.layout.focus(self.buffer)

        answer = "y" if result else "n"
        # Store the confirmation exactly as it appeared in the terminal:
        # no artificial indentation before the question.
        history_message = message.rstrip("\n") + f"\nDo you want to continue? [Y/n] {answer}\n\n"
        self.transcript.append(history_message)
        self.app.invalidate()
        return result

    def set_confirmation_choice(self, result: bool) -> None:
        if not self._confirmation_active:
            return
        self._confirmation_choice = result

        # Show the selected answer in the same input field. Keep only one
        # character so Y/N behaves like a terminal confirmation prompt.
        self.confirmation_buffer.text = "y" if result else "n"
        self.confirmation_buffer.cursor_position = 1
        self._cursor_blink.mark_typing()
        self.app.invalidate()

    def resolve_confirmation(self, result: bool) -> None:
        if not self._confirmation_active:
            return
        self._confirmation_result = result
        self._confirmation_event.set()
        self.app.invalidate()

    def submit_text(self, event, text: str) -> None:
        """Submit work in a background thread without leaving the UI."""
        if not text.strip() or (self._submit_thread and self._submit_thread.is_alive()):
            return

        self.buffer.reset()
        self.pasted_texts.clear()
        self.paste_counter = 0

        def worker() -> None:
            should_exit = False
            try:
                if self.on_submit is not None:
                    should_exit = bool(self.on_submit(text))
            except KeyboardInterrupt:
                self.transcript.append(f"\n{GRAY}[Interrupted]{RESET}\n\n")
            except Exception as exc:
                self.transcript.append(f"\n{RED}[Error]{RESET} {exc}\n\n")
            finally:
                self._submit_thread = None
                if should_exit:
                    try:
                        self.app.exit()
                    except Exception:
                        pass
                else:
                    self.app.invalidate()

        self._submit_thread = threading.Thread(
            target=worker,
            daemon=True,
            name="echo-turn-worker",
        )
        self._submit_thread.start()

    def run(self) -> None:
        """Run one persistent prompt application for the entire REPL session."""
        self._cursor_blink.start(self.app)
        try:
            self.app.run()
        finally:
            self._cursor_blink.stop()
            sys.stdout.write(CURSOR_HIDE)
            sys.stdout.flush()

    def prompt(self) -> str:
        """Compatibility wrapper for one-shot callers."""
        self.buffer.reset()
        self.pasted_texts.clear()
        self.paste_counter = 0
        self._cursor_blink.start(self.app)
        try:
            result = self.app.run()
        finally:
            self._cursor_blink.stop()
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
        # Interactive /clear is handled by the persistent UI transcript.
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
# Banner / live random logo animation
# ============================================================

LOGO_ANIMATION_FPS = 30


def _banner_info_lines(config: Optional[EchoConfig]) -> list[str]:
    info_lines = [
        f"{BOLD}{WHITE}Echo CLI 0.1.0{RESET}",
        f"{GRAY}Local AI Support Assistant{RESET}",
    ]
    if config is not None:
        info_lines.append(
            f"{GRAY}{config.model} (via Ollama @ {config.ollama_url}){RESET}"
        )
        info_lines.append(
            f"{GRAY}Workspace: {config.workspace_root}{RESET}"
        )
    else:
        info_lines.append(f"{GRAY}Qwen 2.5 3B (via Ollama){RESET}")
        info_lines.append(f"{GRAY}workspace: ./workspace{RESET}")

    logo_height = len(ECHO_LOGO)
    return info_lines + [""] * max(logo_height - len(info_lines), 0)


class LogoAnimator:
    """
    Live random logo animator.

    The animation is rendered INSIDE prompt_toolkit instead of writing to
    stdout from a background thread. This keeps the logo animation and the
    input prompt synchronized and prevents the animation from corrupting the
    prompt_toolkit layout.

    The available effects are:
      - rotation     : validated ECHO_LOGO_FRAMES
      - particles    : procedural particle dispersion/convergence
      - materialize  : progressive construction of the logo
      - glitch       : short video-artifact bursts

    A different effect is selected after each completed effect, with no
    immediate repetition. The thread runs continuously for the lifetime of
    the input application.
    """

    def __init__(
        self,
        config: Optional[EchoConfig] = None,
        fps: int = LOGO_ANIMATION_FPS,
        enabled: bool = True,
    ):
        self.config = config
        self.fps = fps
        self.enabled = enabled
        self.logo_height = len(ECHO_LOGO)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._app: Optional[Application] = None
        self._last_animation: Optional[str] = None
        self._frame_lock = threading.Lock()
        self._frame: list[str] = list(ECHO_LOGO)
        self._points = [
            (x, y)
            for y, row in enumerate(ECHO_LOGO)
            for x, char in enumerate(row)
            if char == ":"
        ]

    def _compose(self, logo_frame: list[str]) -> list[str]:
        padded_info = _banner_info_lines(self.config)
        return [
            f"  {WHITE}{logo_line}{RESET}   {info_line}"
            for logo_line, info_line in zip(logo_frame, padded_info)
        ]

    def current_lines(self) -> list[str]:
        with self._frame_lock:
            frame = list(self._frame)
        return self._compose(frame)

    def set_frame(self, frame: list[str]) -> None:
        with self._frame_lock:
            self._frame = list(frame)
        app = self._app
        if app is not None:
            try:
                app.invalidate()
            except Exception:
                pass

    def _sleep_frame(self, fps: Optional[int] = None) -> bool:
        return not self._stop_event.wait(1.0 / (fps or self.fps))

    def _render(self, frame: list[str]) -> bool:
        if self._stop_event.is_set():
            return False
        self.set_frame(frame)
        return True

    def _render_colored(self, frame: list[str]) -> bool:
        return self._render(frame)

    def _rotation(self) -> None:
        for frame in ECHO_LOGO_FRAMES:
            if not self._render_colored(frame):
                return
            if not self._sleep_frame():
                return

    def _particle_frame(self, positions, brightness: float) -> list[str]:
        width = len(ECHO_LOGO[0])
        grid = [[" " for _ in range(width)] for _ in range(self.logo_height)]
        for x, y in positions:
            ix, iy = round(x), round(y)
            if 0 <= ix < width and 0 <= iy < self.logo_height:
                grid[iy][ix] = ":"
        color = WHITE if brightness >= 0.80 else GRAY if brightness >= 0.55 else DARK
        return [
            "".join(f"{color}:{RESET}" if char == ":" else char for char in row)
            for row in grid
        ]

    def _particles(self) -> None:
        rng = random.Random()
        cx = sum(x for x, _ in self._points) / len(self._points)
        cy = sum(y for _, y in self._points) / len(self._points)
        data = []

        for x, y in self._points:
            data.append({
                "x": x,
                "y": y,
                "angle": math.atan2(y - cy, x - cx),
                "radius": rng.uniform(3.0, 7.0),
                "speed": rng.uniform(0.7, 1.3),
                "phase": rng.uniform(0.0, math.tau),
                "drift": rng.uniform(0.10, 0.40),
            })

        for _ in range(8):
            if self._stop_event.is_set():
                return
            if not self._render(ECHO_LOGO):
                return
            if not self._sleep_frame():
                return

        for i in range(26):
            if self._stop_event.is_set():
                return
            t = i / 25.0
            t = t * t * (3.0 - 2.0 * t)
            positions = []

            for p in data:
                radius = p["radius"] * t
                tx = (
                    cx
                    + math.cos(p["angle"]) * radius
                    + math.sin(i * 0.17 + p["phase"]) * p["drift"] * t
                )
                ty = (
                    cy
                    + math.sin(p["angle"]) * radius * 0.65
                    + math.cos(i * 0.13 + p["phase"]) * p["drift"] * t
                )
                positions.append((
                    p["x"] * (1.0 - t) + tx * t,
                    p["y"] * (1.0 - t) + ty * t,
                ))

            if not self._render(self._particle_frame(positions, 1.0 - t * 0.25)):
                return
            if not self._sleep_frame():
                return

        for i in range(42):
            if self._stop_event.is_set():
                return
            positions = []

            for p in data:
                angle = p["angle"] + i * 0.025 * p["speed"]
                radius = p["radius"] + math.sin(i * 0.10 + p["phase"]) * 0.45
                positions.append((
                    cx
                    + math.cos(angle) * radius
                    + math.sin(i * 0.07 + p["phase"]) * 0.20,
                    cy
                    + math.sin(angle) * radius * 0.65
                    + math.cos(i * 0.11 + p["phase"]) * 0.20,
                ))

            if not self._render(self._particle_frame(positions, 0.85)):
                return
            if not self._sleep_frame():
                return

        for i in range(30):
            if self._stop_event.is_set():
                return
            t = i / 29.0
            t = t * t * (3.0 - 2.0 * t)
            positions = []

            for p in data:
                angle = p["angle"] + 42 * 0.025 * p["speed"]
                radius = p["radius"] * (1.0 - t)
                ox = cx + math.cos(angle) * radius
                oy = cy + math.sin(angle) * radius * 0.65
                positions.append((
                    ox * (1.0 - t) + p["x"] * t,
                    oy * (1.0 - t) + p["y"] * t,
                ))

            if not self._render(self._particle_frame(positions, 0.55 + t * 0.45)):
                return
            if not self._sleep_frame():
                return

        for _ in range(8):
            if self._stop_event.is_set():
                return
            if not self._render(ECHO_LOGO):
                return
            if not self._sleep_frame():
                return

    def _materialize(self) -> None:
        rng = random.Random()
        points = self._points[:]
        rng.shuffle(points)
        current = set()
        empty = [" " * len(row) for row in ECHO_LOGO]

        for _ in range(8):
            if self._stop_event.is_set():
                return
            if not self._render(empty):
                return
            if not self._sleep_frame():
                return

        for x, y in points:
            if self._stop_event.is_set():
                return
            current.add((x, y))
            frame = []

            for yy, row in enumerate(ECHO_LOGO):
                frame.append("".join(
                    ":" if char == ":" and (xx, yy) in current else " "
                    for xx, char in enumerate(row)
                ))

            if not self._render(frame):
                return
            if not self._sleep_frame():
                return

        for _ in range(12):
            if self._stop_event.is_set():
                return
            if not self._render(ECHO_LOGO):
                return
            if not self._sleep_frame():
                return

    def _glitch_frame(self):
        result = []
        shift = random.choice([-3, -2, -1, 1, 2, 3])

        for row in ECHO_LOGO:
            if random.random() < 0.45:
                result.append(row)
                continue

            shifted = [" "] * len(row)

            for x, char in enumerate(row):
                new_x = x + shift
                if 0 <= new_x < len(row) and char == ":":
                    shifted[new_x] = ":"

            for _ in range(random.randint(1, 3)):
                x = random.randrange(len(row))
                if shifted[x] == ":":
                    shifted[x] = random.choice(["·", "░", "▒", "▓"])

            if random.random() < 0.25:
                start = random.randrange(len(row))
                length = random.randint(1, 4)
                for x in range(start, min(start + length, len(row))):
                    shifted[x] = " "

            result.append("".join(shifted))

        glitch_colors = (
            "[38;2;255;70;70m",
            "[38;2;0;240;255m",
            "[38;2;255;0;180m",
            "[38;2;90;140;255m",
        )

        colored = []
        for row in result:
            line = []
            for char in row:
                if char in (":", "·", "░", "▒", "▓") and random.random() < 0.38:
                    line.append(f"{random.choice(glitch_colors)}{char}{WHITE}")
                else:
                    line.append(char)
            colored.append("".join(line))

        return colored

    def _glitch(self) -> None:
        for _ in range(18):
            if self._stop_event.is_set():
                return
            if not self._render(ECHO_LOGO):
                return
            if not self._sleep_frame():
                return

        for _ in range(7):
            if self._stop_event.is_set():
                return

            for _ in range(random.randint(7, 15)):
                if self._stop_event.is_set():
                    return
                if not self._render(ECHO_LOGO):
                    return
                if not self._sleep_frame():
                    return

            for _ in range(random.randint(2, 5)):
                if self._stop_event.is_set():
                    return
                if not self._render(self._glitch_frame()):
                    return
                if not self._sleep_frame():
                    return

        for _ in range(12):
            if self._stop_event.is_set():
                return
            if not self._render(ECHO_LOGO):
                return
            if not self._sleep_frame():
                return

    # --------------------------------------------------------
    # Tetris speed-build
    # --------------------------------------------------------

    def _tetris_frame(self, locked: set[tuple[int, int]],
                      falling: list[tuple[int, int, str]] | None = None) -> list[str]:
        """
        Render one Tetris frame inside the same 8-line logo viewport.

        The falling pieces use a fixed velocity trail:

            ·
            ░
            ▒
            ▓

        The actual target cells are taken from ECHO_LOGO and are locked as
        ':' when the piece reaches its destination. This keeps the speed-run
        effect while guaranteeing that the final image is the real logo.
        """
        grid = [[" " for _ in row] for row in ECHO_LOGO]

        for x, y in locked:
            if 0 <= y < self.logo_height and 0 <= x < len(ECHO_LOGO[0]):
                grid[y][x] = ":"

        if falling:
            for x, y, char in falling:
                if 0 <= y < self.logo_height and 0 <= x < len(ECHO_LOGO[0]):
                    grid[y][x] = char

        rendered = []
        for row in grid:
            rendered.append(
                "".join(
                    f"{WHITE}:{RESET}" if c == ":" else
                    f"{WHITE}▓{RESET}" if c == "▓" else
                    f"{GRAY}▒{RESET}" if c == "▒" else
                    f"{GRAY}░{RESET}" if c == "░" else
                    f"{GRAY}·{RESET}" if c == "·" else
                    c
                    for c in row
                )
            )
        return rendered

    def _tetris(self) -> None:
        """
        Build the real logo from bottom to top, one tiny piece at a time.

        The fall is intentionally extremely fast. The eye should mostly see
        a stream of fragments assembling the logo, rather than recognizable
        tetromino shapes.
        """
        rng = random.Random()

        # Exact target cells, grouped by row. We process rows bottom -> top.
        rows = {
            y: [x for x, char in enumerate(row) if char == ":"]
            for y, row in enumerate(ECHO_LOGO)
        }

        locked: set[tuple[int, int]] = set()

        # Start from empty space.
        empty = [" " * len(row) for row in ECHO_LOGO]
        if not self._render(empty):
            return
        if not self._sleep_frame(24):
            return

        for y in range(self.logo_height - 1, -1, -1):
            xs = rows[y][:]
            rng.shuffle(xs)

            while xs:
                # Tiny fragments: usually 1–2 cells, occasionally 3.
                size = rng.choices([1, 2, 3], weights=[50, 40, 10])[0]
                size = min(size, len(xs))
                piece_xs = xs[:size]
                del xs[:size]

                # Keep the piece exactly shaped as the selected logo cells.
                min_x = min(piece_xs)
                max_x = max(piece_xs)

                # Each selected cell has its own vertical trail.
                start_y = -4 - rng.randint(0, 2)

                # Fast drop. A few intermediate frames are enough for speed.
                for step in range(4):
                    current_y = round(
                        start_y + (y - start_y) * ((step + 1) / 4.0)
                    )

                    falling: list[tuple[int, int, str]] = []

                    # Piece head.
                    for px in piece_xs:
                        falling.append((px, current_y, "▓"))

                    # Speed trail is ALWAYS ordered from weak -> strong.
                    # Only one vertical trail per piece-column is shown.
                    for px in range(min_x, max_x + 1):
                        if px not in piece_xs:
                            continue
                        for offset, trail_char in enumerate(("░", "▒", "▓"), 1):
                            trail_y = current_y - offset
                            if 0 <= trail_y < self.logo_height:
                                falling.append((px, trail_y, trail_char))

                    if not self._render(self._tetris_frame(locked, falling)):
                        return
                    if not self._sleep_frame(45):
                        return

                # Instant lock into the REAL logo.
                for px in piece_xs:
                    locked.add((px, y))

                if not self._render(self._tetris_frame(locked)):
                    return
                if not self._sleep_frame(120):
                    return

        # Exact final logo.
        if not self._render(ECHO_LOGO):
            return
        self._sleep_frame(18)

    def _choose_animation(self):
        # All effects remain implemented, but only these two are currently
        # enabled in the random scheduler.
        animations = [
            ("rotation", self._rotation),
            ("particles", self._particles),
            # Disabled for now; keep implementations available for later.
            # ("materialize", self._materialize),
            # ("glitch", self._glitch),
            # ("tetris", self._tetris),
        ]
        available = [
            item for item in animations
            if item[0] != self._last_animation
        ]
        name, callback = random.choice(available or animations)
        self._last_animation = name
        return callback

    def _run(self):
        # Start from the stable logo, then continuously select effects.
        self.set_frame(ECHO_LOGO)

        while not self._stop_event.is_set():
            animation = self._choose_animation()
            animation()

            if self._stop_event.wait(random.uniform(0.08, 0.18)):
                return

    def start(self, app: Application) -> None:
        if not self.enabled:
            self.set_frame(ECHO_LOGO)
            return

        if self._thread and self._thread.is_alive():
            return

        self._app = app
        self._stop_event.clear()
        self._last_animation = None
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="echo-logo-animation",
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

        if self._thread:
            self._thread.join(timeout=1.0)

        self._thread = None
        self._app = None
        self.set_frame(ECHO_LOGO)


def print_banner(config: Optional[EchoConfig] = None):
    """Static banner for non-interactive/non-TTY use."""
    for line in LogoAnimator(config, enabled=False).current_lines():
        # Keep ANSI already embedded in the composed banner.
        print(line)


# ============================================================
# Streaming UI
# ============================================================

class StreamingUI:
    def __init__(self, verbose: bool = True, workspace_root: Optional[Path] = None):
        self.verbose = verbose
        self.workspace_root = workspace_root
        self.spinner = Spinner("Loading")
        self._printing = False
        self._after_tool = False
        self._gap = False
        self.transcript: Optional[LiveTranscript] = None
        self.confirmation_handler = None
        self.spinner.set_status_sink(self._set_status)

    def attach_transcript(self, transcript: LiveTranscript) -> None:
        self.transcript = transcript

    def _append(self, text: str) -> None:
        if self.transcript is not None:
            self.transcript.append(text)
        else:
            sys.stdout.write(text)
            sys.stdout.flush()

    def _set_status(self, text: str) -> None:
        if self.transcript is not None:
            self.transcript.set_status(text)
        else:
            if text:
                sys.stdout.write(f"\r{GRAY}{text}{RESET}")
            else:
                sys.stdout.write("\r\033[K")
            sys.stdout.flush()

    def reset(self):
        self.spinner.stop()
        self._printing = False
        self._after_tool = False
        self._gap = False

    def _resolve_tool_path(self, value) -> Optional[Path]:
        if not isinstance(value, str) or not value.strip():
            return None
        path = Path(value).expanduser()
        if not path.is_absolute() and self.workspace_root is not None:
            path = self.workspace_root / path
        try:
            return path.resolve()
        except OSError:
            return path

    @staticmethod
    def _format_bytes(value: int) -> str:
        value = max(0, int(value))
        units = ("B", "KB", "MB", "GB")
        size = float(value)
        for unit in units:
            if size < 1024 or unit == units[-1]:
                if unit == "B":
                    return f"{int(size)} B"
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{int(value)} B"

    def confirm_tool(self, tool_name: str, arguments: dict) -> bool:
        """Build an apt-style operation summary and ask in the UI footer."""
        self.spinner.stop()
        self._printing = False

        if tool_name == "write_file":
            path_text = arguments.get("path", "?")
            content = arguments.get("content", "")
            payload_size = len(content.encode("utf-8")) if isinstance(content, str) else 0
            path = self._resolve_tool_path(path_text)
            exists = path.exists() if path is not None else False

            if exists and path is not None and path.is_file():
                try:
                    old_size = path.stat().st_size
                except OSError:
                    old_size = 0
                delta = payload_size - old_size
                action = "modified"
                if delta > 0:
                    disk_line = f"After this operation, {self._format_bytes(delta)} of additional disk space will be used."
                elif delta < 0:
                    disk_line = f"After this operation, {self._format_bytes(-delta)} of disk space will be freed."
                else:
                    disk_line = "After this operation, no additional disk space will be used."
                summary = "0 created, 1 modified, 0 deleted."
            else:
                action = "created"
                disk_line = f"After this operation, {self._format_bytes(payload_size)} of additional disk space will be used."
                summary = "0 modified, 1 created, 0 deleted."

            message = (
                f"\nThe following file will be {action}:\n"
                f"{path_text}\n"
                f"{summary}\n"
                f"{disk_line}\n"
            )

        elif tool_name == "edit_file":
            path_text = arguments.get("path", "?")
            old_text = arguments.get("old_text", "")
            new_text = arguments.get("new_text", "")
            old_bytes = len(old_text.encode("utf-8")) if isinstance(old_text, str) else 0
            new_bytes = len(new_text.encode("utf-8")) if isinstance(new_text, str) else 0
            delta = new_bytes - old_bytes

            if delta > 0:
                disk_line = f"After this operation, {self._format_bytes(delta)} of additional disk space will be used."
            elif delta < 0:
                disk_line = f"After this operation, {self._format_bytes(-delta)} of disk space will be freed."
            else:
                disk_line = "After this operation, no additional disk space will be used."

            message = (
                f"\nThe following file will be modified:\n"
                f"{path_text}\n"
                "0 created, 1 modified, 0 deleted.\n"
                f"{disk_line}\n"
            )

        elif tool_name == "delete_file":
            path_text = arguments.get("path", "?")
            path = self._resolve_tool_path(path_text)
            try:
                size = path.stat().st_size if path is not None and path.is_file() else 0
            except OSError:
                size = 0

            message = (
                f"\nThe following file will be deleted:\n"
                f"{path_text}\n"
                "0 modified, 0 created, 1 deleted.\n"
                f"After this operation, {self._format_bytes(size)} of disk space will be freed.\n"
            )

        else:
            message = (
                f"The following operation will be performed:\n"
                f"{tool_name}\n\n"
                "Do you want to continue? [Y/n]\n"
            )

        handler = getattr(self, "confirmation_handler", None)
        if handler is not None:
            return handler(message)

        self._append(message)
        return False

    def _lead(self) -> str:
        if self._gap:
            self._gap = False
            return ""
        return "\n"

    def handle_event(self, event_type: str, data: dict):
        if event_type == "run_started":
            self.reset()
        elif event_type == "iteration_started":
            if self._after_tool:
                self._append("\n")
                self._after_tool = False
            self._gap = True
            self.spinner.start("Loading")
        elif event_type == "content_delta":
            self.spinner.stop()
            if not self._printing:
                self._append(self._lead())
                self._append(f"{GRAY}> {RESET}")
                self._printing = True
            self._append(f"{WHITE}{data['delta']}{RESET}")
        elif event_type == "tool_call_received":
            self.spinner.stop()
            if self.verbose:
                self._after_tool = True
                tool_name = data["tool_name"]
                arguments = data["arguments"]
                try:
                    if isinstance(arguments, str):
                        arguments = json.loads(arguments)
                    formatted_arguments = json.dumps(arguments, ensure_ascii=False, indent=2)
                except (TypeError, ValueError, json.JSONDecodeError):
                    formatted_arguments = str(arguments)
                self._append(f"{self._lead()}{GRAY}[Tool Call]{RESET}\n")
                self._append(f"{GRAY}{tool_name}({formatted_arguments}){RESET}\n")
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
                    self._append(f"\n{header}\n{WHITE}{result_text}{RESET}\n")
                else:
                    self._append(f"\n{header} {GRAY}-> {result_text}{RESET}\n")
        elif event_type == "workspace_reloaded":
            self._append(f"\n{GRAY}[Tools reloaded -> {data['workspace_root']}]{RESET}\n")
        elif event_type == "error":
            self.spinner.stop()
            self._append(f"\n{BOLD}{RED}[Error]{RESET} {data.get('error')}\n")
        elif event_type in ("run_completed", "max_iterations_reached"):
            self.spinner.stop()
            if self._printing:
                self._append("\n")
                self._printing = False


# ============================================================
# Main
# ============================================================

def main():
    # Clear previous visible content and scrollback once at startup.
    # Never clear on Ctrl+C or normal REPL exit.
    sys.stdout.write("\033[3J\033[2J\033[H")
    sys.stdout.flush()

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

    ui = StreamingUI(verbose=verbose, workspace_root=config.workspace_root)
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
    # One Application lives for the whole session. The banner is therefore
    # never re-created below the conversation after a model turn.
    echo_input = EchoInput(config, animate_logo=(not args.quiet))
    ui.attach_transcript(echo_input.transcript)
    ui.confirmation_handler = echo_input.request_confirmation
    last_stats: Optional[str] = None

    def handle_submitted_input(user_input: str) -> bool:
        nonlocal last_stats

        if not user_input.strip():
            return False

        # Persistent UI commands stay inside the transcript. They must never
        # clear/repaint the terminal underneath the prompt_toolkit layout.
        if user_input == "\x00/clear":
            ui.reset()
            echo_input.transcript.clear()
            echo_input.last_display = ""
            return False

        if user_input == "\x00/help":
            echo_input.transcript.append(f"\n{_show_help()}\n\n")
            echo_input.last_display = ""
            return False

        # A new turn always starts at the newest part of the conversation.
        echo_input.transcript_scroll.scroll_to_end()

        shown = echo_input.last_display or user_input
        echo_lines = shown.splitlines() or [shown]
        ui._append(f"{GRAY}> {echo_lines[0]}{RESET}\n")
        for extra in echo_lines[1:]:
            ui._append(f"{GRAY}  {extra}{RESET}\n")
        # Keep a blank line between the user's prompt and the model response.
        ui._append("\n")
        echo_input.last_display = ""

        if user_input == "\x00/new":
            orchestrator.reset_history()
            ui._append(f"\n{GREEN}Conversation cleared.{RESET}\n\n")
            return False

        if user_input.startswith("\x00"):
            if user_input == "\x00/exit":
                ui._append("Goodbye!\n")
                return True
            should_continue, _model_changed, workspace_changed = execute_slash_command(
                user_input, config, last_stats
            )
            if not should_continue:
                ui._append("Goodbye!\n")
                return True
            if workspace_changed:
                orchestrator.reload_workspace()
            return False

        if user_input.lower() in ("exit", "quit"):
            ui._append("Goodbye!\n")
            return True

        try:
            result = orchestrator.run(user_input)
            ui._append("\n")
            if verbose:
                last_stats = format_stats(result)
                ui._append(f"{GRAY}[{last_stats}]{RESET}\n\n")
        except KeyboardInterrupt:
            ui.reset()
            ui._append(f"\n{GRAY}[Interrupted]{RESET}\n\n")
        return False

    echo_input.on_submit = handle_submitted_input

    def final_exit_screen() -> None:
        """Leave only the final banner/divider and a clean Exiting message."""
        try:
            # Stop animation first so its background thread cannot repaint the
            # screen while we are constructing the final exit state.
            echo_input.stop_logo_animation()

            # Clear only the visible screen (not terminal scrollback), then
            # redraw the last stable banner and its separator.
            sys.stdout.write("\033[2J\033[H")
            sys.stdout.write("\n".join(echo_input.logo_animator.current_lines()))
            sys.stdout.write("\n")
            sys.stdout.write(
                "─" * echo_input.get_content_width()
            )
            sys.stdout.write("\n\nExiting...\n\n")
            sys.stdout.write(CURSOR_RESTORE)
            sys.stdout.flush()
        except Exception:
            # The terminal restore must never turn an exit into another error.
            try:
                sys.stdout.write(CURSOR_RESTORE)
                sys.stdout.flush()
            except Exception:
                pass

    try:
        echo_input.run()
    except KeyboardInterrupt:
        final_exit_screen()
        return
    except EOFError:
        final_exit_screen()
        return
    else:
        # /exit and /quit close prompt_toolkit normally (without an exception).
        final_exit_screen()
        return
    finally:
        # Safe no-op if final_exit_screen already stopped the animation.
        echo_input.stop_logo_animation()
        sys.stdout.write(CURSOR_RESTORE)
        sys.stdout.flush()


if __name__ == "__main__":
    main()
    