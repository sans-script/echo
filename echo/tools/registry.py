"""Tool registry and validation system for Echo."""

import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class ToolExecutionRecord:
    """Record of a tool call executed on the host."""
    call_id: str
    tool_name: str
    arguments: Dict[str, Any]
    raw_arguments: str
    result: str
    success: bool
    duration_seconds: float
    error: Optional[str] = None


class ToolRegistry:
    """Registry for tools exposed to Echo language model."""

    def __init__(self, max_output_chars: int = 8000):
        self._tools: Dict[str, Dict[str, Any]] = {}
        self.max_output_chars = max_output_chars
        self.execution_history: List[ToolExecutionRecord] = []

    def register(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        handler: Callable[..., Any],
        requires_confirmation: bool = False,
    ) -> None:
        """Register a tool with its OpenAI-compatible JSON schema and host handler."""
        if name in self._tools:
            raise ValueError(f"Tool '{name}' is already registered.")

        self._tools[name] = {
            "name": name,
            "description": description,
            "parameters": parameters,
            "handler": handler,
            "requires_confirmation": requires_confirmation,
        }

    def requires_confirmation(self, tool_name: str) -> bool:
        """Return whether a tool requires explicit user confirmation."""
        tool = self._tools.get(tool_name)
        return bool(tool and tool.get("requires_confirmation", False))

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Return OpenAI-compatible tool specifications list."""
        definitions = []
        for name, data in self._tools.items():
            definitions.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": data["description"],
                    "parameters": data["parameters"],
                }
            })
        return definitions

    def validate_and_parse_args(
        self, tool_name: str, raw_arguments: str
    ) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        """Validate that arguments are valid JSON and conform to the tool parameter schema."""
        if tool_name not in self._tools:
            available = ", ".join(self._tools.keys())
            return False, None, f"Unknown tool '{tool_name}'. Available tools: [{available}]"

        # 1. Parse JSON
        if isinstance(raw_arguments, dict):
            args = raw_arguments
        else:
            try:
                args = json.loads(raw_arguments) if raw_arguments else {}
            except json.JSONDecodeError as e:
                return False, None, f"Invalid JSON in tool arguments: {e}"

        if not isinstance(args, dict):
            return False, None, f"Arguments must be a JSON object, got {type(args).__name__}"

        schema = self._tools[tool_name]["parameters"]
        properties = schema.get("properties", {})
        required = schema.get("required", [])

        # 2. Check required parameters
        missing = [p for p in required if p not in args]
        if missing:
            return False, None, f"Missing required parameter(s): {', '.join(missing)}"

        # 3. Reject parameters that are not declared by the schema.
        unknown = [p for p in args if p not in properties]
        if unknown:
            return False, None, (
                f"Unknown parameter(s): {', '.join(unknown)}. "
                f"Allowed parameters: {', '.join(properties.keys()) or '(none)'}"
            )

        # 4. Coerce values to the types declared by the schema.
        type_mapping = {
            "string": (str,),
            "integer": (int,),
            "number": (int, float),
            "boolean": (bool,),
            "array": (list,),
            "object": (dict,),
        }

        def coerce_value(param_name: str, value: Any, expected_type: str) -> Any:
            """Safely coerce common JSON/model-generated values to the schema type."""
            if expected_type not in type_mapping:
                return value

            # Already the correct type.
            # bool must be handled separately because bool is an int subclass.
            if expected_type == "integer":
                if isinstance(value, bool):
                    raise ValueError("got boolean value")
                if isinstance(value, int):
                    return value
            elif expected_type == "number":
                if isinstance(value, bool):
                    raise ValueError("got boolean value")
                if isinstance(value, (int, float)):
                    return value
            elif expected_type == "boolean":
                if isinstance(value, bool):
                    return value
            elif expected_type == "string":
                if isinstance(value, str):
                    return value
            elif expected_type == "array":
                if isinstance(value, list):
                    return value
            elif expected_type == "object":
                if isinstance(value, dict):
                    return value

            # Models sometimes emit scalar values as strings.
            if isinstance(value, str):
                stripped = value.strip()

                if expected_type == "integer":
                    try:
                        # Do not accept "3.5" as an integer.
                        parsed = int(stripped)
                        if stripped not in {str(parsed), f"+{parsed}", f"-{abs(parsed)}"}:
                            raise ValueError
                        return parsed
                    except ValueError:
                        raise ValueError(
                            f"could not convert string {value!r} to integer"
                        )

                if expected_type == "number":
                    try:
                        parsed = json.loads(stripped)
                    except json.JSONDecodeError:
                        raise ValueError(
                            f"could not convert string {value!r} to number"
                        )

                    if isinstance(parsed, bool) or not isinstance(parsed, (int, float)):
                        raise ValueError(
                            f"could not convert string {value!r} to number"
                        )
                    return parsed

                if expected_type == "boolean":
                    lowered = stripped.lower()
                    if lowered == "true":
                        return True
                    if lowered == "false":
                        return False
                    raise ValueError(
                        f"could not convert string {value!r} to boolean; "
                        "expected 'true' or 'false'"
                    )

                if expected_type in ("array", "object"):
                    try:
                        parsed = json.loads(stripped)
                    except json.JSONDecodeError:
                        raise ValueError(
                            f"could not parse string {value!r} as JSON "
                            f"{expected_type}"
                        )

                    expected_python_type = list if expected_type == "array" else dict
                    if not isinstance(parsed, expected_python_type):
                        raise ValueError(
                            f"JSON value must be a {expected_type}"
                        )
                    return parsed

            raise ValueError(
                f"expected type '{expected_type}', got '{type(value).__name__}'"
            )

        coerced_args = {}

        for param_name, val in args.items():
            expected_type_str = properties[param_name].get("type")

            if expected_type_str and expected_type_str in type_mapping:
                try:
                    coerced_args[param_name] = coerce_value(
                        param_name,
                        val,
                        expected_type_str,
                    )
                except ValueError as e:
                    return False, None, (
                        f"Invalid value for parameter '{param_name}': {e}"
                    )
            else:
                coerced_args[param_name] = val

        return True, coerced_args, None

    def execute(self, call_id: str, tool_name: str, raw_arguments: str) -> ToolExecutionRecord:
        """Validate and execute a tool call, returning the execution record."""
        start_time = time.perf_counter()

        valid, parsed_args, err_msg = self.validate_and_parse_args(tool_name, raw_arguments)
        if not valid:
            elapsed = time.perf_counter() - start_time
            record = ToolExecutionRecord(
                call_id=call_id,
                tool_name=tool_name,
                arguments={},
                raw_arguments=str(raw_arguments),
                result=f"Error: {err_msg}",
                success=False,
                duration_seconds=elapsed,
                error=err_msg,
            )
            self.execution_history.append(record)
            return record

        handler = self._tools[tool_name]["handler"]
        try:
            raw_result = handler(**parsed_args)
            result_str = str(raw_result)
            # Truncate if tool output is excessively long to prevent token blowout
            if len(result_str) > self.max_output_chars:
                result_str = (
                    result_str[: self.max_output_chars]
                    + f"\n... [Output truncated; total length: {len(result_str)} characters]"
                )
            success = True
            error_str = None
        except Exception as e:
            result_str = f"Error executing tool '{tool_name}': {type(e).__name__}: {e}"
            success = False
            error_str = str(e)

        elapsed = time.perf_counter() - start_time
        record = ToolExecutionRecord(
            call_id=call_id,
            tool_name=tool_name,
            arguments=parsed_args,
            raw_arguments=str(raw_arguments),
            result=result_str,
            success=success,
            duration_seconds=elapsed,
            error=error_str,
        )
        self.execution_history.append(record)
        return record
