"""Echo Orchestrator and Tool-Calling Loop."""

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .client import OllamaClient, OllamaClientError
from .config import EchoConfig
from .tools.filesystem import register_filesystem_tools
from .tools.registry import ToolExecutionRecord, ToolRegistry


@dataclass
class OrchestratorResult:
    """The result of an Echo orchestrator run."""

    user_prompt: str
    final_response: str
    iterations: int
    tool_executions: List[ToolExecutionRecord] = field(default_factory=list)
    messages: List[Dict[str, Any]] = field(default_factory=list)
    total_duration_seconds: float = 0.0
    model_name: str = ""
    stopped_reason: str = "completed"
    error_message: Optional[str] = None

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class EchoOrchestrator:
    """Core orchestrator executing the tool-calling loop."""

    # True quando o terminal ja mostra o resultado das tools ao usuario.
    tool_output_visible: bool = True

    def __init__(
        self,
        config: Optional[EchoConfig] = None,
        client: Optional[OllamaClient] = None,
        registry: Optional[ToolRegistry] = None,
        on_event: Optional[
            Callable[[str, Dict[str, Any]], None]
        ] = None,
        confirm_tool: Optional[
            Callable[[str, Dict[str, Any]], bool]
        ] = None,
    ):
        self.config = config or EchoConfig()
        self.history: List[Dict[str, Any]] = []

        self.client = client or OllamaClient(
            base_url=self.config.ollama_url,
            timeout=self.config.request_timeout,
        )

        self.registry = registry or ToolRegistry(
            max_output_chars=self.config.max_tool_output_chars
        )

        self.on_event = on_event or (
            lambda event, data: None
        )

        self.confirm_tool = confirm_tool or (
            lambda tool_name, arguments: False
        )

        self.filesystem_handler = register_filesystem_tools(
            self.registry,
            self.config.workspace_root,
        )

    def reload_workspace(self) -> None:
        """
        Re-register filesystem tools with the current workspace_root.

        Called when the user runs /workspace <path> so that subsequent
        tool calls operate on the new directory.

        The old FilesystemToolHandler is discarded and a fresh one is
        built against self.config.workspace_root. All filesystem tool
        handlers (write_file, read_file, list_directory, ...) are
        re-bound to the new handler instance.

        Ref: Codex (app_server_session.rs re-initializes per-workspace
        tool handlers when cwd changes) and Agy (jetski/cli/store/store.go
        ApplyProjectPermissionGrants reloads permissions after a
        workspace switch).
        """
        # Names of filesystem tools registered by register_filesystem_tools.
        # We remove them so the next register call re-binds to the new
        # FilesystemToolHandler instead of the stale one.
        # Only the three filesystem tools actually registered by
        # register_filesystem_tools() in tools/filesystem.py.
        FS_TOOL_NAMES = (
            "write_file",
            "read_file",
            "list_directory",
            "tree",
            "edit_file",
            "search_files",
            "find_files",
        )

        for name in FS_TOOL_NAMES:
            self.registry._tools.pop(name, None)

        # Re-register against the updated workspace_root.
        self.filesystem_handler = register_filesystem_tools(
            self.registry,
            self.config.workspace_root,
        )

        self.history = []

        self._emit(
            "workspace_reloaded",
            {
                "workspace_root": str(self.config.workspace_root),
            },
        )

    def _emit(
        self,
        event_type: str,
        data: Dict[str, Any],
    ) -> None:
        """Emit an orchestrator event."""
        try:
            self.on_event(event_type, data)
        except Exception:
            pass

    def reset_history(self) -> None:
        """Forget the conversation so far."""
        self.history = []

    def _trim_history(self, msgs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Keep the last few turns and shrink bulky tool outputs."""
        max_turns = 6
        max_tool_chars = 2000

        trimmed: List[Dict[str, Any]] = []
        for m in msgs:
            m = dict(m)
            if m.get("role") == "tool":
                content = m.get("content") or ""
                if len(content) > max_tool_chars:
                    m["content"] = content[:max_tool_chars] + "\n... [truncated in history]"
            trimmed.append(m)

        user_idx = [i for i, m in enumerate(trimmed) if m.get("role") == "user"]
        if len(user_idx) > max_turns:
            trimmed = trimmed[user_idx[-max_turns]:]
        return trimmed

    def _workspace_context(self) -> str:
        """Snapshot of the workspace, appended to the system prompt each turn."""
        try:
            tree_fn = getattr(self.filesystem_handler, "tree", None)
            if tree_fn is not None:
                snapshot = tree_fn(".", 2, False, 60, True)
            else:
                snapshot = self.filesystem_handler.list_directory(".")
        except Exception:
            return ""

        return (
            "\n\nWorkspace root: " + str(self.config.workspace_root) + "\n"
            "Current workspace snapshot (use these exact paths and do not guess file names):\n"
            + snapshot
        )

    def run(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
    ) -> OrchestratorResult:
        """Run one turn, keeping the conversation history between calls."""
        result = self._run(user_prompt, system_prompt)
        if result.stopped_reason == "completed":
            # messages[0] is the system prompt, which is rebuilt every turn.
            self.history = self._trim_history(result.messages[1:])
        return result

    def _run(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
    ) -> OrchestratorResult:
        """Run the tool-calling loop."""

        start_time = time.perf_counter()

        tool_defs = self.registry.get_tool_definitions()

        sys_prompt = (
            system_prompt
            or self.config.system_prompt
        )
        sys_prompt = sys_prompt + self._workspace_context()

        messages: List[Dict[str, Any]] = (
            [{"role": "system", "content": sys_prompt}]
            + list(self.history)
            + [{"role": "user", "content": user_prompt}]
        )

        self._emit(
            "run_started",
            {
                "prompt": user_prompt,
                "model": self.config.model,
                "tools_count": len(tool_defs),
            },
        )

        all_tool_executions: List[
            ToolExecutionRecord
        ] = []

        iteration = 0
        call_signatures_history: List[str] = []

        force_text_response = False
        empty_response_retries = 0

        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0

        while iteration < self.config.max_iterations:
            iteration += 1

            self._emit(
                "iteration_started",
                {
                    "iteration": iteration,
                },
            )

            current_tools = (
                None
                if force_text_response
                else (
                    tool_defs
                    if tool_defs
                    else None
                )
            )

            msg: Optional[Dict[str, Any]] = None
            content_buffer = ""

            try:
                for event in self.client.chat_completion_stream(
                    model=self.config.model,
                    messages=messages,
                    tools=current_tools,
                    temperature=self.config.temperature,
                ):
                    if event["type"] == "content":
                        # On the first iteration, buffer content instead of
                        # emitting it live. This avoids flashing the raw
                        # tool-call JSON on screen before the fallback
                        # converts it into an actual tool_calls list.
                        # Ref: Agy (jetski/cli/steps/steps.go buffers the
                        # model payload before deciding activity vs text).
                        content_buffer += event["delta"]

                    elif event["type"] == "tool_call_delta":
                        self._emit(
                            "tool_call_delta",
                            {
                                "delta": event["delta"],
                                "iteration": iteration,
                            },
                        )

                    elif event["type"] == "done":
                        msg = event["message"]

                        usage = event.get("usage") or {}

                        prompt_tokens += usage.get(
                            "prompt_tokens",
                            0,
                        )

                        completion_tokens += usage.get(
                            "completion_tokens",
                            0,
                        )

                        total_tokens += usage.get(
                            "total_tokens",
                            0,
                        )

                if msg is None:
                    raise OllamaClientError(
                        "Stream ended without a final assistant message"
                    )

                # On the first iteration, if we buffered content and it
                # was NOT a tool call (structured or JSON fallback), flush
                # it as a normal response delta so the user sees it.
                if content_buffer:
                    pending_tool_calls = msg.get("tool_calls")

                    if not pending_tool_calls:
                        stripped = content_buffer.strip()
                        if stripped.startswith("```") and stripped.endswith("```"):
                            stripped = stripped.strip("`").strip()
                            if stripped.lower().startswith("json"):
                                stripped = stripped[4:].strip()
                        looks_like_json_tool_call = (
                            stripped.startswith("{")
                            and stripped.endswith("}")
                            and '"name"' in stripped
                        )

                        if not looks_like_json_tool_call:
                            self._emit(
                                "content_delta",
                                {
                                    "delta": content_buffer,
                                    "iteration": iteration,
                                },
                            )

            except OllamaClientError as exc:
                total_duration = (
                    time.perf_counter()
                    - start_time
                )

                self._emit(
                    "error",
                    {
                        "error": str(exc),
                        "iteration": iteration,
                    },
                )

                return OrchestratorResult(
                    user_prompt=user_prompt,
                    final_response="",
                    iterations=iteration,
                    tool_executions=all_tool_executions,
                    messages=messages,
                    total_duration_seconds=total_duration,
                    model_name=self.config.model,
                    stopped_reason="error",
                    error_message=str(exc),
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                )

            messages.append(msg)

            tool_calls = msg.get("tool_calls")

            # ── Fallback: some models (qwen2.5-coder, older llama variants)
            # do not return structured tool_calls via the Ollama API, even
            # when they understand the tool invocation. Instead, they emit
            # the call as a JSON string inside `content`, e.g.:
            #
            #   {"name": "list_directory", "arguments": {"path": "."}}
            #
            # This fallback detects that pattern, wraps it into the same
            # shape the rest of the loop expects, and strips the JSON from
            # `content` so it is not duplicated in the final response.
            #
            # Ref: Codex (chat_composer.rs fallback for non-native tool
            # calling models) and Agy (jetski/cli/steps/steps.go
            # ExtractToolInfo / DeriveActivityLabel handling of the
            # structured JSON payload returned by models without native
            # function calling).
            if not tool_calls:
                content = (msg.get("content") or "").strip()

                # Unwrap markdown code fences (```json ... ```) if present.
                if content.startswith("```") and content.endswith("```"):
                    inner = content.strip("`").strip()
                    if inner.lower().startswith("json"):
                        inner = inner[4:].strip()
                    content = inner

                # Only attempt the fallback if the content looks like a
                # single JSON object (starts with `{` and ends with `}`).
                if content.startswith("{") and content.endswith("}"):
                    try:
                        parsed = json.loads(content)

                        candidate_name = (
                            parsed.get("name")
                            or parsed.get("tool")
                            or parsed.get("tool_name")
                        )

                        candidate_args = (
                            parsed.get("arguments")
                            or parsed.get("args")
                            or parsed.get("parameters")
                            or {}
                        )

                        if (
                            isinstance(candidate_name, str)
                            and candidate_name
                            and isinstance(candidate_args, (dict, list))
                        ):
                            # Re-wrap into the standard tool_calls shape.
                            tool_calls = [
                                {
                                    "id": f"call_{uuid.uuid4().hex[:8]}",
                                    "function": {
                                        "name": candidate_name,
                                        "arguments": json.dumps(
                                            candidate_args
                                        ),
                                    },
                                }
                            ]

                            # Strip the JSON from the visible content so
                            # the model output is not echoed twice.
                            msg["content"] = ""

                            self._emit(
                                "tool_call_detected_fallback",
                                {
                                    "tool_name": candidate_name,
                                    "arguments": json.dumps(
                                        candidate_args
                                    ),
                                },
                            )

                    except (json.JSONDecodeError, TypeError, ValueError):
                        # Not a JSON tool call — treat as a normal text
                        # response. No action needed.
                        pass

            if not tool_calls or force_text_response:
                final_content = msg.get(
                    "content",
                    "",
                )

                # Some local models occasionally finish a stream without
                # producing either text or tool calls. Retry once instead of
                # silently ending the user's turn.
                if (
                    not tool_calls
                    and not force_text_response
                    and not final_content.strip()
                    and empty_response_retries < 1
                ):
                    empty_response_retries += 1

                    # Remove the empty assistant message so the retry sees
                    # exactly the same conversation state as the first call.
                    if messages and messages[-1] is msg:
                        messages.pop()

                    self._emit(
                        "empty_response_retry",
                        {
                            "iteration": iteration,
                            "retry": empty_response_retries,
                        },
                    )
                    continue

                if not final_content.strip():
                    final_content = (
                        "O modelo não retornou uma resposta. "
                        "Tente novamente."
                    )

                total_duration = (
                    time.perf_counter()
                    - start_time
                )

                self._emit(
                    "run_completed",
                    {
                        "final_response": final_content,
                        "iterations": iteration,
                        "tool_executions": len(
                            all_tool_executions
                        ),
                        "duration": total_duration,
                    },
                )

                return OrchestratorResult(
                    user_prompt=user_prompt,
                    final_response=final_content,
                    iterations=iteration,
                    tool_executions=all_tool_executions,
                    messages=messages,
                    total_duration_seconds=total_duration,
                    model_name=self.config.model,
                    stopped_reason="completed",
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                )

            consecutive_repeats = 0

            for tool_call in tool_calls:
                call_id = (
                    tool_call.get("id")
                    or f"call_{uuid.uuid4().hex[:8]}"
                )

                function = tool_call.get(
                    "function",
                    {},
                )

                tool_name = function.get(
                    "name",
                    "",
                )

                raw_args = function.get(
                    "arguments",
                    "{}",
                )

                try:
                    _parsed_args = json.loads(raw_args) if raw_args.strip() else {}
                    if tool_name == "list_directory" and _parsed_args.get("path") in (None, "", "."):
                        _parsed_args = {}
                    _norm_args = json.dumps(_parsed_args, sort_keys=True)
                except (json.JSONDecodeError, AttributeError):
                    _norm_args = raw_args.strip()

                signature = f"{tool_name}:{_norm_args}"

                if (
                    call_signatures_history
                    and call_signatures_history[-1]
                    == signature
                ):
                    consecutive_repeats += 1

                call_signatures_history.append(
                    signature
                )

                self._emit(
                    "tool_call_received",
                    {
                        "call_id": call_id,
                        "tool_name": tool_name,
                        "arguments": raw_args,
                    },
                )

                # Ask for explicit user confirmation before executing
                # tools marked as requiring confirmation.
                if self.registry.requires_confirmation(tool_name):
                    try:
                        parsed_for_confirmation = json.loads(raw_args) if raw_args.strip() else {}
                    except (json.JSONDecodeError, AttributeError):
                        parsed_for_confirmation = {"_raw_arguments": raw_args}

                    confirmed = self.confirm_tool(
                        tool_name,
                        parsed_for_confirmation,
                    )

                    self._emit(
                        "tool_confirmation",
                        {
                            "call_id": call_id,
                            "tool_name": tool_name,
                            "arguments": raw_args,
                            "confirmed": confirmed,
                        },
                    )

                    if not confirmed:
                        result_content = (
                            f"Tool '{tool_name}' execution was cancelled by the user."
                        )

                        execution = ToolExecutionRecord(
                            call_id=call_id,
                            tool_name=tool_name,
                            arguments=(
                                parsed_for_confirmation
                                if isinstance(parsed_for_confirmation, dict)
                                else {}
                            ),
                            raw_arguments=str(raw_args),
                            result=result_content,
                            success=False,
                            duration_seconds=0.0,
                            error="User denied confirmation.",
                        )

                        all_tool_executions.append(execution)

                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": call_id,
                                "name": tool_name,
                                "content": result_content,
                            }
                        )

                        continue

                execution = self.registry.execute(
                    call_id=call_id,
                    tool_name=tool_name,
                    raw_arguments=raw_args,
                )

                all_tool_executions.append(
                    execution
                )

                result_content = execution.result

                if (
                    self.tool_output_visible
                    and execution.success
                    and tool_name in ("tree", "list_directory")
                ):
                    result_content += (
                        "\n[Note: this listing is already displayed to the user. "
                        "Do not repeat it; answer with one short sentence.]"
                    )

                if consecutive_repeats >= 1:
                    result_content += (
                        "\n[Notice: You have already "
                        "executed this tool with the "
                        "same arguments. Please "
                        "synthesize your final response.]"
                    )

                self._emit(
                    "tool_call_executed",
                    {
                        "call_id": call_id,
                        "tool_name": tool_name,
                        "success": execution.success,
                        "result": execution.result,
                        "duration": execution.duration_seconds,
                    },
                )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": tool_name,
                        "content": result_content,
                    }
                )

            if consecutive_repeats >= 1:
                force_text_response = True

        total_duration = (
            time.perf_counter()
            - start_time
        )

        self._emit(
            "max_iterations_reached",
            {
                "iterations": iteration,
            },
        )

        return OrchestratorResult(
            user_prompt=user_prompt,
            final_response=(
                "Maximum tool-calling iterations "
                "reached before completing."
            ),
            iterations=iteration,
            tool_executions=all_tool_executions,
            messages=messages,
            total_duration_seconds=total_duration,
            model_name=self.config.model,
            stopped_reason="max_iterations_reached",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )