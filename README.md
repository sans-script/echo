# Echo — Local AI Support Assistant

Echo is a lightweight, local AI support assistant designed to handle inexpensive, repetitive, or context-processing tasks as part of an experimental multi-agent engineering environment.

Powered by **Qwen 2.5 3B Instruct** running locally through Ollama, Echo pairs small-parameter local inference with a controlled, secure host-side tool execution layer.

---

## Architecture

```text
               ┌───────────────────────┐
               │         User          │
               │   (CLI / REPL / API)  │
               └───────────┬───────────┘
                           │
                           ▼
               ┌───────────────────────┐
               │         Echo          │
               │     Orchestrator      │
               └───────────┬───────────┘
                           │
                           ▼
               ┌───────────────────────┐
               │     Ollama Server     │
               │ Qwen 2.5 3B Instruct  │
               └───────────┬───────────┘
                           │
                     function_call
                           │
                           ▼
               ┌───────────────────────┐
               │     Tool Registry     │
               │ (Validator & Monitor) │
               └───────────┬───────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
          read_file    write_file   list_directory
              │            │            │
              └────────────┼────────────┘
                           │
                           ▼
                      tool result
                           │
                           ▼
                     Qwen 2.5 3B
                           │
                           ▼
                     final response
```

---

## Features

- **Direct Ollama Integration**: Communicates directly with the Ollama OpenAI-compatible `/v1/chat/completions` API without heavy framework overhead.
- **Strict Host-Side Tool Sandboxing**: All filesystem operations are locked to a designated workspace directory (`./workspace` by default), preventing directory traversal (`../`) and unauthorized access to system files (`/etc/passwd`).
- **Comprehensive Tool Validation**: Validates tool names, parameter schemas, argument types, and JSON parsing before any host execution.
- **Loop Prevention & Hallucination Guard**: Detects redundant tool calls and forces text completion if the model repeats identical calls, preventing runaway loops common in small models.
- **Fast Local Inference**: Average tool invocation + response turnaround of 3–7 seconds on local hardware.

---

## Installation & Requirements

- Python 3.10+
- `requests` package (`pip install requests`)
- Ollama running locally at `http://localhost:11434` with model `qwen2.5:3b-instruct`:
  ```bash
  ollama run qwen2.5:3b-instruct
  ```

---

## Usage

### 1. Run Single Prompt
```bash
python3 run_echo.py "Create a file named test-echo.txt containing exactly: TOOL CALL OK"
```

### 2. Interactive REPL
```bash
python3 run_echo.py
```

### 3. Check Health
```bash
python3 run_echo.py --check-health
```

### 4. Custom Workspace
```bash
python3 run_echo.py --workspace /path/to/workspace "List all files"
```

---

## Testing

Run unit tests:
```bash
python3 -m unittest tests/test_tools_unit.py
```

Run live end-to-end integration tests (requires Ollama):
```bash
python3 -m unittest tests/test_live_experiment.py
```

Run all tests:
```bash
python3 -m unittest discover tests -v
```
