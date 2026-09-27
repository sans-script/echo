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
from .logo_frames import ECHO_LOGO, ECHO_LOGO_FRAMES
from .logo_animation import play as play_logo_animation


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
    "/logo":      "Replay the Echo logo animation",
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
            elif cmd == "/logo":
                buffer.reset()
                event.app.exit(result="\x00/logo")
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
            self.logo_window,
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
        output = create_cursorless_output()
        self.app = Application(
            layout=self.layout,
            key_bindings=create_key_bindings(self),
            style=PROMPT_STYLE,
            cursor=CursorShape._NEVER_CHANGE,
            full_screen=False,
            erase_when_done=True,
            output=output,
        )

        # Disable CPR immediately after Application construction.
        # This must happen before app.run() is called.
        self.app.renderer.cpr_support = CPR_Support.NOT_SUPPORTED

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

    def prompt(self) -> str:
        """Run the application and return the typed text."""
        # Clear the buffer BEFORE running the application, so that
        # text from the previous interaction is not reused.
        self.buffer.reset()

        # Clear paste state for the new prompt cycle.
        self.pasted_texts.clear()
        self.paste_counter = 0

        self._cursor_blink.start(self.app)
        try:
            result = self.app.run()
        finally:
            self._cursor_blink.stop()

        # Keep the physical cursor hidden while model output is streamed.
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

    elif payload == "/logo":
        print()
        play_logo_animation("rotation")
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
        animations = [
            ("rotation", self._rotation),
            ("particles", self._particles),
            ("materialize", self._materialize),
            ("glitch", self._glitch),
            ("tetris", self._tetris),
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
    echo_input = EchoInput(config, animate_logo=(not args.quiet))
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
        echo_input.stop_logo_animation()
        # Restore the cursor to visible + default user shape when
        # leaving the REPL. Without this, the terminal may be left
        # with an invisible cursor or a "blinking block" shape.
        sys.stdout.write(CURSOR_RESTORE)
        sys.stdout.flush()


if __name__ == "__main__":
    main()
    