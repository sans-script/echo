"""Configuration management for Echo local assistant."""

import os
from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_SYSTEM_PROMPT = (
    "You are Echo, a lightweight local AI support assistant. "
    "You help with file operations, information extraction, summarization, and planning. "
    "When you need to read, write, or list workspace files, call the appropriate tool. "
    "Once you receive the tool results, synthesize and provide your final response to the user. "
    "Do not call the same tool repeatedly with the same parameters. "
    "If the user's message already contains the text or code to work on, use it directly and do not call tools to find it. "
    "Only read files the user names explicitly or that you found by listing the workspace. "
    "If a file does not exist, say so instead of retrying. "
    "When a tool already displayed a directory tree or file list to the user, do not copy it again; add at most one short sentence. "
    "If the answer is already in the workspace snapshot or in earlier messages, answer directly without calling any tool. "
    "Be concise, direct, and factual."
)


@dataclass
class EchoConfig:
    ollama_url: str = "http://localhost:11434"
    model: str = "qwen2.5-coder"
    workspace_root: Path = field(default_factory=lambda: Path(os.environ.get("ECHO_WORKSPACE", "./workspace")).resolve())
    max_iterations: int = 10
    temperature: float = 0.0
    request_timeout: float = 60.0
    max_tool_output_chars: int = 8000
    system_prompt: str = DEFAULT_SYSTEM_PROMPT

    def __post_init__(self):
        if isinstance(self.workspace_root, str):
            self.workspace_root = Path(self.workspace_root).resolve()
        # Ensure workspace root exists
        self.workspace_root.mkdir(parents=True, exist_ok=True)
