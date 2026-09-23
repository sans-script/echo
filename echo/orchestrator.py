"""Echo Orchestrator and Tool-Calling Loop."""

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

    def __init__(
        self,
        config: Optional[EchoConfig] = None,
        client: Optional[OllamaClient] = None,
        registry: Optional[ToolRegistry] = None,
        on_event: Optional[
            Callable[[str, Dict[str, Any]], None]
        ] = None,
    ):
        self.config = config or EchoConfig()

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

        self.filesystem_handler = register_filesystem_tools(
            self.registry,
            self.config.workspace_root,
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

    def run(
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

        messages: List[Dict[str, Any]] = [
            {
                "role": "system",
                "content": sys_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]

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

            try:
                for event in self.client.chat_completion_stream(
                    model=self.config.model,
                    messages=messages,
                    tools=current_tools,
                    temperature=self.config.temperature,
                ):
                    if event["type"] == "content":
                        self._emit(
                            "content_delta",
                            {
                                "delta": event["delta"],
                                "iteration": iteration,
                            },
                        )

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

            if not tool_calls or force_text_response:
                total_duration = (
                    time.perf_counter()
                    - start_time
                )

                final_content = msg.get(
                    "content",
                    "",
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

                signature = (
                    f"{tool_name}:"
                    f"{raw_args.strip()}"
                )

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

                execution = self.registry.execute(
                    call_id=call_id,
                    tool_name=tool_name,
                    raw_arguments=raw_args,
                )

                all_tool_executions.append(
                    execution
                )

                result_content = execution.result

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