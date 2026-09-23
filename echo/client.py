"""Ollama API client for Echo."""

import json
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple

import requests


class OllamaClientError(Exception):
    """Base exception for Ollama client errors."""
    pass


class OllamaClient:
    """Client for Ollama's OpenAI-compatible and native endpoints."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout: float = 60.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def check_health(
        self,
        required_model: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """Check if Ollama server is responding and if the required model is available."""
        try:
            resp = requests.get(
                f"{self.base_url}/api/tags",
                timeout=10.0,
            )

            if resp.status_code != 200:
                return (
                    False,
                    f"Ollama returned HTTP {resp.status_code}: {resp.text}",
                )

            data = resp.json()
            models = [
                model.get("name", "")
                for model in data.get("models", [])
            ]

            if required_model:
                matching = [
                    model
                    for model in models
                    if model == required_model
                    or model.startswith(f"{required_model}:")
                ]

                if not matching:
                    return (
                        False,
                        f"Model '{required_model}' not found "
                        f"in available models: {models}",
                    )

            return True, "Ollama is reachable and model is available."

        except requests.RequestException as exc:
            return (
                False,
                f"Failed to connect to Ollama at "
                f"{self.base_url}: {exc}",
            )

    def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.0,
    ) -> Tuple[Dict[str, Any], float]:
        """Non-streaming chat completion."""
        url = f"{self.base_url}/v1/chat/completions"

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }

        if tools:
            payload["tools"] = tools

        start_time = time.perf_counter()

        try:
            resp = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=self.timeout,
            )
        except requests.Timeout:
            raise OllamaClientError(
                f"Request to Ollama timed out after {self.timeout}s"
            )
        except requests.RequestException as exc:
            raise OllamaClientError(
                f"Error communicating with Ollama: {exc}"
            )

        elapsed = time.perf_counter() - start_time

        if resp.status_code != 200:
            raise OllamaClientError(
                f"Ollama API returned HTTP {resp.status_code}: {resp.text}"
            )

        try:
            data = resp.json()
            choice = data["choices"][0]
            message = choice["message"]

            if data.get("usage"):
                message["_usage"] = data["usage"]

            return message, elapsed

        except (KeyError, IndexError, json.JSONDecodeError) as exc:
            raise OllamaClientError(
                f"Malformed response from Ollama: {resp.text}"
            ) from exc

    def chat_completion_stream(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.0,
    ) -> Iterator[Dict[str, Any]]:
        """Stream a chat completion from Ollama's OpenAI-compatible endpoint.

        Events:

          {"type": "content", "delta": str}

          {"type": "tool_call_delta", "delta": [...]}

          {
              "type": "done",
              "message": {...},
              "elapsed": float,
              "finish_reason": str|None,
              "usage": dict|None,
          }
        """

        url = f"{self.base_url}/v1/chat/completions"

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }

        if tools:
            payload["tools"] = tools

        start_time = time.perf_counter()

        try:
            resp = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=self.timeout,
                stream=True,
            )
        except requests.Timeout:
            raise OllamaClientError(
                f"Request to Ollama timed out after {self.timeout}s"
            )
        except requests.RequestException as exc:
            raise OllamaClientError(
                f"Error communicating with Ollama: {exc}"
            )

        if resp.status_code != 200:
            raise OllamaClientError(
                f"Ollama API returned HTTP {resp.status_code}: {resp.text}"
            )

        accumulated_content = ""
        accumulated_tool_calls: Dict[int, Dict[str, Any]] = {}

        finish_reason = None
        role = "assistant"
        usage = None

        try:
            for raw_bytes in resp.iter_lines():
                raw_line = raw_bytes.decode("utf-8", errors="replace") if isinstance(raw_bytes, bytes) else raw_bytes
                if not raw_line:
                    continue

                line = raw_line

                if line.startswith("data:"):
                    line = line[len("data:"):].strip()

                if line == "[DONE]":
                    break

                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if chunk.get("usage"):
                    usage = chunk["usage"]

                choices = chunk.get("choices") or []

                if not choices:
                    continue

                choice = choices[0]
                delta = choice.get("delta", {}) or {}

                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]

                if delta.get("role"):
                    role = delta["role"]

                content_piece = delta.get("content")

                if content_piece:
                    accumulated_content += content_piece

                    yield {
                        "type": "content",
                        "delta": content_piece,
                    }

                tool_call_deltas = delta.get("tool_calls")

                if tool_call_deltas:
                    for tool_call in tool_call_deltas:
                        index = tool_call.get("index", 0)

                        entry = accumulated_tool_calls.setdefault(
                            index,
                            {
                                "id": tool_call.get("id")
                                or f"call_{index}",
                                "type": "function",
                                "function": {
                                    "name": "",
                                    "arguments": "",
                                },
                            },
                        )

                        if tool_call.get("id"):
                            entry["id"] = tool_call["id"]

                        function = tool_call.get("function") or {}

                        if function.get("name"):
                            entry["function"]["name"] += function["name"]

                        if function.get("arguments"):
                            entry["function"]["arguments"] += (
                                function["arguments"]
                            )

                    yield {
                        "type": "tool_call_delta",
                        "delta": tool_call_deltas,
                    }

        except requests.RequestException as exc:
            raise OllamaClientError(
                f"Error while streaming from Ollama: {exc}"
            )

        elapsed = time.perf_counter() - start_time

        message: Dict[str, Any] = {
            "role": role,
            "content": accumulated_content,
        }

        if accumulated_tool_calls:
            message["tool_calls"] = [
                accumulated_tool_calls[index]
                for index in sorted(accumulated_tool_calls)
            ]

        yield {
            "type": "done",
            "message": message,
            "elapsed": elapsed,
            "finish_reason": finish_reason,
            "usage": usage,
        }