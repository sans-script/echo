"""Filesystem tools for Echo with sandbox boundary enforcement."""

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

        return f"Successfully wrote {len(encoded)} bytes to '{rel_path}'."

    def read_file(self, path: str, max_bytes: int = 65536) -> str:
        """Read text content from a file within the workspace."""
        target_path, rel_path = self.resolve_path(path)

        if not target_path.exists():
            raise FileNotFoundError(f"File '{rel_path}' does not exist.")

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
            raise FileNotFoundError(f"Directory '{rel_path}' does not exist.")

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

    return handler
