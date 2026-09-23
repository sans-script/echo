"""Echo - Lightweight local AI assistant."""

from .config import EchoConfig
from .client import OllamaClient, OllamaClientError
from .orchestrator import EchoOrchestrator, OrchestratorResult
from .tools.registry import ToolRegistry, ToolExecutionRecord
from .tools.filesystem import FilesystemToolHandler, FilesystemSandboxError

__version__ = "0.1.0"
__all__ = [
    "EchoConfig",
    "OllamaClient",
    "OllamaClientError",
    "EchoOrchestrator",
    "OrchestratorResult",
    "ToolRegistry",
    "ToolExecutionRecord",
    "FilesystemToolHandler",
    "FilesystemSandboxError",
]
