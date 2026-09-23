"""Filesystem tools for Echo with sandbox boundary enforcement."""

import difflib
import fnmatch
import os
from pathlib import Path
from typing import List, Optional, Tuple


class FilesystemSandboxError(Exception):
    """Raised when a filesystem operation violates sandbox boundaries or safety rules."""
    pass


class FilesystemToolHandler:
    """Provides sandboxed filesystem operations within a designated workspace root."""

    SYSTEM_ROOT_DIRS = {"etc", "root", "sys", "proc", "dev", "var", "bin", "usr", "boot", "home"}

    def __init__(self, workspace_root: Path):
        self.workspace_root = Path(workspace_root).resolve()
        self.workspace_root.mkdir(parents=True, exist_ok=True)

    def resolve_path(self, user_path: str) -> Tuple[Path, str]:
        """Safely resolve a path within the workspace root.

        Prevents path traversal attacks and outside filesystem access.
        Returns tuple of (resolved_path, relative_display_path).
        """
        if not user_path or not user_path.strip():
            raise FilesystemSandboxError("Path cannot be empty.")

        cleaned = user_path.strip()

        # Check for obvious traversal tokens before any processing
        raw_parts = Path(cleaned).parts

        # If user passed an absolute path
        if os.path.isabs(cleaned):
            target_path = Path(cleaned).resolve()
            # If the absolute path is already inside the workspace root, allow it
            if self._is_within_workspace(target_path):
                rel_path = target_path.relative_to(self.workspace_root).as_posix()
                return target_path, rel_path if rel_path != "." else "./"

            # Check if this looks like a system directory (e.g. /etc, /root, /var)
            first_component = raw_parts[1] if len(raw_parts) > 1 else ""
            if first_component in self.SYSTEM_ROOT_DIRS:
                raise FilesystemSandboxError(
                    f"Access denied: Path '{cleaned}' attempts to access system directory outside workspace."
                )

            # If the model used a virtual root path like '/test-echo.txt', treat it relative to workspace
            relative_candidate = cleaned.lstrip("/\\")
            resolved = (self.workspace_root / relative_candidate).resolve()
            if not self._is_within_workspace(resolved):
                raise FilesystemSandboxError(
                    f"Access denied: Path '{cleaned}' resolves outside workspace sandbox."
                )
            rel = resolved.relative_to(self.workspace_root).as_posix()
            return resolved, rel

        # Relative path handling
        resolved = (self.workspace_root / cleaned).resolve()
        if not self._is_within_workspace(resolved):
            raise FilesystemSandboxError(
                f"Access denied: Path '{cleaned}' traverses outside workspace sandbox."
            )

        rel = resolved.relative_to(self.workspace_root).as_posix()
        return resolved, rel if rel != "." else "./"

    def _is_within_workspace(self, path: Path) -> bool:
        """Check if path is within or equal to workspace_root."""
        try:
            path.relative_to(self.workspace_root)
            return True
        except ValueError:
            return False

    def write_file(self, path: str, content: str) -> str:
        """Write content to a file at the specified path within the workspace."""
        target_path, rel_path = self.resolve_path(path)

        # Ensure parent directories exist
        target_path.parent.mkdir(parents=True, exist_ok=True)

        # Write content
        encoded = content.encode("utf-8")
        target_path.write_bytes(encoded)

        return (
            f"Successfully wrote {len(encoded)} bytes to '{rel_path}'. "
            f"Location: {target_path}"
        )

    def read_file(self, path: str, max_bytes: int = 65536) -> str:
        """Read text content from a file within the workspace."""
        target_path, rel_path = self.resolve_path(path)

        if not target_path.exists():
            raise FileNotFoundError(f"File '{rel_path}' does not exist." + self._similar_hint(target_path))

        if target_path.is_dir():
            raise IsADirectoryError(f"'{rel_path}' is a directory, not a file. Use list_directory instead.")

        file_size = target_path.stat().st_size
        with open(target_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read(max_bytes)

        if file_size > max_bytes:
            content += f"\n... [File truncated: showing first {max_bytes} of {file_size} bytes]"

        return content

    def list_directory(self, path: str = ".") -> str:
        """List files and subdirectories in the specified workspace directory."""
        target_path, rel_path = self.resolve_path(path)

        if not target_path.exists():
            raise FileNotFoundError(f"Directory '{rel_path}' does not exist." + self._similar_hint(target_path))

        if not target_path.is_dir():
            raise NotADirectoryError(f"'{rel_path}' is a file, not a directory. Use read_file instead.")

        entries = sorted(target_path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        if not entries:
            return f"(Directory '{rel_path}' is empty)"

        lines = [f"Contents of '{rel_path}':"]
        for entry in entries:
            if entry.is_dir():
                lines.append(f"  [DIR]  {entry.name}/")
            else:
                size = entry.stat().st_size
                lines.append(f"  [FILE] {entry.name} ({size} bytes)")

        return "\n".join(lines)

    def tree(
        self,
        path: str = ".",
        max_depth: int = 2,
        show_hidden: bool = False,
        max_entries: int = 200,
        show_sizes: bool = False,
    ) -> str:
        """Render a directory as a tree (like the `tree` command)."""
        target_path, rel_path = self.resolve_path(path)

        if not target_path.exists():
            raise FileNotFoundError(f"Directory '{rel_path}' does not exist." + self._similar_hint(target_path))
        if not target_path.is_dir():
            raise NotADirectoryError(f"'{rel_path}' is a file, not a directory. Use read_file instead.")

        # Small models often send numbers/booleans as strings.
        try:
            max_depth = max(1, min(int(max_depth), 6))
        except (TypeError, ValueError):
            max_depth = 2
        try:
            max_entries = max(10, min(int(max_entries), 500))
        except (TypeError, ValueError):
            max_entries = 200
        if isinstance(show_hidden, str):
            show_hidden = show_hidden.strip().lower() in ("true", "1", "yes", "sim")
        if isinstance(show_sizes, str):
            show_sizes = show_sizes.strip().lower() in ("true", "1", "yes", "sim")

        ignored = {"__pycache__", "node_modules", ".git", ".venv", "venv", ".mypy_cache", ".pytest_cache"}
        lines = ["." if rel_path == "./" else rel_path]
        stats = {"dirs": 0, "files": 0, "truncated": False}

        def is_link(entry: Path) -> bool:
            try:
                if entry.is_symlink():
                    return True
                is_junction = getattr(entry, "is_junction", None)
                return bool(is_junction and is_junction())
            except OSError:
                return True

        def walk(directory: Path, prefix: str, depth: int) -> None:
            try:
                entries = sorted(directory.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
            except OSError:
                lines.append(f"{prefix}└── [access denied]")
                return

            visible = [
                e for e in entries
                if e.name not in ignored and (show_hidden or not e.name.startswith("."))
            ]

            for index, entry in enumerate(visible):
                if stats["truncated"]:
                    return
                if stats["dirs"] + stats["files"] >= max_entries:
                    lines.append(f"{prefix}└── ... (+{len(visible) - index} more)")
                    stats["truncated"] = True
                    return

                is_last = index == len(visible) - 1
                branch = "└── " if is_last else "├── "

                if entry.is_dir():
                    stats["dirs"] += 1
                    if is_link(entry):
                        lines.append(f"{prefix}{branch}{entry.name}/ (link)")
                        continue
                    lines.append(f"{prefix}{branch}{entry.name}/")
                    if depth < max_depth:
                        walk(entry, prefix + ("    " if is_last else "│   "), depth + 1)
                else:
                    stats["files"] += 1
                    size_note = ""
                    if show_sizes:
                        try:
                            size_note = f" ({entry.stat().st_size} bytes)"
                        except OSError:
                            pass
                    lines.append(f"{prefix}{branch}{entry.name}{size_note}")

        walk(target_path, "", 1)

        note = "truncated" if stats["truncated"] else f"depth {max_depth}"
        hidden_note = "" if show_hidden else ", hidden omitted"
        lines.append(f"\n{stats['dirs']} directories, {stats['files']} files ({note}{hidden_note})")
        return "\n".join(lines)


    _IGNORED_DIRS = {"__pycache__", "node_modules", ".git", ".venv", "venv", ".mypy_cache", ".pytest_cache"}

    def _similar_hint(self, target_path: Path) -> str:
        # Suggests existing paths when the requested one does not exist.
        name = target_path.name
        stem = target_path.stem.lower()
        hits: List[str] = []
        visited = 0

        for root, dirs, files in os.walk(self.workspace_root):
            dirs[:] = [d for d in dirs if d not in self._IGNORED_DIRS and not d.startswith(".")]
            for entry in files + dirs:
                visited += 1
                low = entry.lower()
                if low == name.lower() or (len(stem) >= 3 and stem in low):
                    rel = (Path(root) / entry).relative_to(self.workspace_root).as_posix()
                    if rel not in hits:
                        hits.append(rel)
            if visited > 5000 or len(hits) >= 5:
                break

        if not hits and target_path.parent.is_dir():
            try:
                siblings = [e.name for e in target_path.parent.iterdir()]
            except OSError:
                siblings = []
            for match in difflib.get_close_matches(name, siblings, n=3, cutoff=0.5):
                hits.append((target_path.parent / match).relative_to(self.workspace_root).as_posix())

        if hits:
            return " Did you mean: " + ", ".join(hits[:5]) + "?"
        return " Use list_directory or tree to see which files exist."

    def edit_file(self, path: str, old_text: str, new_text: str) -> str:
        # Replaces exactly one occurrence of old_text with new_text.
        target_path, rel_path = self.resolve_path(path)

        if not target_path.exists():
            raise FileNotFoundError(f"File '{rel_path}' does not exist." + self._similar_hint(target_path))
        if target_path.is_dir():
            raise IsADirectoryError(f"'{rel_path}' is a directory, not a file.")
        if not old_text:
            raise ValueError("old_text cannot be empty. Use write_file to create or overwrite a file.")
        if new_text is None:
            new_text = ""

        try:
            content = target_path.read_bytes().decode("utf-8")
        except UnicodeDecodeError:
            raise ValueError(f"'{rel_path}' is not a UTF-8 text file.")

        # Keep Windows line endings working when the model sends plain \n.
        if "\r\n" in content and "\r\n" not in old_text:
            old_text = old_text.replace("\n", "\r\n")
            new_text = new_text.replace("\n", "\r\n")

        count = content.count(old_text)
        if count == 0:
            raise ValueError(
                f"old_text was not found in '{rel_path}'. "
                "Use read_file and copy the exact text, including spaces and indentation."
            )
        if count > 1:
            raise ValueError(
                f"old_text appears {count} times in '{rel_path}'. "
                "Include more surrounding lines so it matches exactly once."
            )

        updated = content.replace(old_text, new_text, 1)
        target_path.write_bytes(updated.encode("utf-8"))
        return (
            f"Edited '{rel_path}': replaced 1 occurrence "
            f"({len(old_text)} -> {len(new_text)} chars). "
            f"Location: {target_path}"
        )

    def search_files(
        self,
        query: str,
        path: str = ".",
        glob: str = "*",
        max_results: int = 30,
        ignore_case: bool = True,
    ) -> str:
        # Text search (like grep) inside workspace files.
        target_path, rel_path = self.resolve_path(path)

        if not query:
            raise ValueError("query cannot be empty.")
        if not target_path.exists():
            raise FileNotFoundError(f"Path '{rel_path}' does not exist." + self._similar_hint(target_path))

        try:
            max_results = max(1, min(int(max_results), 100))
        except (TypeError, ValueError):
            max_results = 30
        if isinstance(ignore_case, str):
            ignore_case = ignore_case.strip().lower() not in ("false", "0", "no", "nao")
        glob = (glob or "*").strip().lower()

        needle = query.lower() if ignore_case else query
        matches: List[str] = []
        files_scanned = 0
        scan_limit = 2000
        truncated = False
        scan_limit_hit = False

        def iter_files():
            if target_path.is_file():
                yield target_path
                return
            for root, dirs, files in os.walk(target_path):
                dirs[:] = [d for d in dirs if d not in self._IGNORED_DIRS and not d.startswith(".")]
                for f in sorted(files):
                    if f.startswith("."):
                        continue
                    if fnmatch.fnmatch(f.lower(), glob):
                        yield Path(root) / f

        for file_path in iter_files():
            if files_scanned >= scan_limit:
                scan_limit_hit = True
                break
            try:
                if file_path.stat().st_size > 1_000_000:
                    continue
                with open(file_path, "rb") as fh:
                    head = fh.read(1024)
                if b"\x00" in head:
                    continue
                text = file_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            files_scanned += 1
            rel_file = file_path.relative_to(self.workspace_root).as_posix()
            for lineno, line in enumerate(text.splitlines(), 1):
                hay = line.lower() if ignore_case else line
                if needle in hay:
                    snippet = line.strip()
                    if len(snippet) > 160:
                        snippet = snippet[:160] + "..."
                    matches.append(f"{rel_file}:{lineno}: {snippet}")
                    if len(matches) >= max_results:
                        truncated = True
                        break
            if truncated:
                break

        note = " (scan limit reached)" if scan_limit_hit else ""
        if not matches:
            return f"No matches for '{query}' in {files_scanned} file(s) under '{rel_path}'{note}."
        header = f"{len(matches)} match(es)" + (" (limit reached)" if truncated else "") + f" under '{rel_path}'{note}:"
        return header + "\n" + "\n".join(matches)

    def find_files(self, pattern: str, path: str = ".", max_results: int = 50) -> str:
        # Finds files and folders by name (wildcards allowed).
        target_path, rel_path = self.resolve_path(path)

        if not pattern or not pattern.strip():
            raise ValueError("pattern cannot be empty.")
        if not target_path.exists():
            raise FileNotFoundError(f"Directory '{rel_path}' does not exist." + self._similar_hint(target_path))
        if not target_path.is_dir():
            raise NotADirectoryError(f"'{rel_path}' is a file, not a directory.")

        try:
            max_results = max(1, min(int(max_results), 200))
        except (TypeError, ValueError):
            max_results = 50

        pat = pattern.strip().lower()
        if not any(ch in pat for ch in "*?["):
            pat = f"*{pat}*"

        results: List[str] = []
        visited = 0
        truncated = False

        for root, dirs, files in os.walk(target_path):
            dirs[:] = sorted(d for d in dirs if d not in self._IGNORED_DIRS and not d.startswith("."))
            entries = [(d, True) for d in dirs] + [(f, False) for f in sorted(files) if not f.startswith(".")]
            for entry, is_dir in entries:
                visited += 1
                if fnmatch.fnmatch(entry.lower(), pat):
                    rel = (Path(root) / entry).relative_to(self.workspace_root).as_posix()
                    results.append(rel + ("/" if is_dir else ""))
                    if len(results) >= max_results:
                        truncated = True
                        break
            if truncated or visited > 20000:
                break

        if not results:
            return f"No files or folders matching '{pattern}' under '{rel_path}'."
        header = f"{len(results)} result(s)" + (" (limit reached)" if truncated else "") + f" for '{pattern}' under '{rel_path}':"
        return header + "\n" + "\n".join(results)



def register_filesystem_tools(registry, workspace_root: Path) -> FilesystemToolHandler:
    """Register all filesystem tools with the given ToolRegistry."""
    handler = FilesystemToolHandler(workspace_root)

    registry.register(
        name="write_file",
        description="Write text content to a file at the specified path within the workspace.",
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The path to the file to create or overwrite.",
                },
                "content": {
                    "type": "string",
                    "description": "The text content to write to the file.",
                },
            },
            "required": ["path", "content"],
        },
        handler=handler.write_file,
        requires_confirmation=True,
    )

    registry.register(
        name="read_file",
        description="Read the text content of a file at the specified path within the workspace.",
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The path of the file to read.",
                },
            },
            "required": ["path"],
        },
        handler=handler.read_file,
    )

    registry.register(
        name="list_directory",
        description="List files and folders in a directory within the workspace.",
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The directory path to list (defaults to '.' for root).",
                },
            },
            "required": [],
        },
        handler=handler.list_directory,
    )

    registry.register(
        name="tree",
        description=(
            "Show a directory structure as a tree, like the `tree` command. "
            "Prefer this over list_directory when the user asks for the structure, "
            "the tree, or nested folders."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory to show (defaults to '.').",
                },
                "max_depth": {
                    "type": "integer",
                    "description": "How many levels to show (default 2, max 6).",
                },
                "show_hidden": {
                    "type": "boolean",
                    "description": "Include entries starting with '.' (default false).",
                },
                "show_sizes": {
                    "type": "boolean",
                    "description": "Show file sizes in bytes (default false).",
                },
            },
            "required": [],
        },
        handler=handler.tree,
    )

    registry.register(
        name="edit_file",
        description=(
            "Replace one exact piece of text in an existing file. Read the file first and copy "
            "old_text exactly. Prefer this over write_file for small changes."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "The file to edit."},
                "old_text": {"type": "string", "description": "Exact text to replace. Must match exactly once."},
                "new_text": {"type": "string", "description": "Text that replaces old_text."},
            },
            "required": ["path", "old_text", "new_text"],
        },
        handler=handler.edit_file,
        requires_confirmation=True,
    )

    registry.register(
        name="search_files",
        description=(
            "Search for text inside workspace files (like grep). Returns file:line: text for each match. "
            "Use it to find where something is defined or mentioned."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Text to look for."},
                "path": {"type": "string", "description": "Folder or file to search (defaults to '.')."},
                "glob": {"type": "string", "description": "Only search files whose name matches, e.g. '*.py' (default '*')."},
                "max_results": {"type": "integer", "description": "Maximum matches to return (default 30)."},
            },
            "required": ["query"],
        },
        handler=handler.search_files,
    )

    registry.register(
        name="find_files",
        description=(
            "Find files and folders by name. Wildcards are allowed, e.g. '*.py'; "
            "plain text matches any name containing it."
        ),
        parameters={
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Name or wildcard pattern, e.g. 'config' or '*.txt'."},
                "path": {"type": "string", "description": "Folder to search (defaults to '.')."},
                "max_results": {"type": "integer", "description": "Maximum results to return (default 50)."},
            },
            "required": ["pattern"],
        },
        handler=handler.find_files,
    )

    return handler
