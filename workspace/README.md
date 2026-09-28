# Echo — Local AI Support Assistant

Echo is a lightweight, local AI support assistant designed to handle inexpensive, repetitive, or context-processing tasks as part of an experimental multi-agent engineering environment.

Powered by **Qwen 2.5 3B Instruct** running locally through **Ollama**, Echo combines small-parameter local inference with a controlled host-side tool execution layer.

The project is intentionally lightweight: the model handles reasoning and tool selection, while Echo remains responsible for validation, filesystem access, user confirmation, execution, and terminal interaction.

---

## Architecture

```text
                         ┌───────────────────────┐
                         │         User          │
                         │      CLI / REPL       │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │         Echo          │
                         │      CLI / UI         │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │     Orchestrator      │
                         │  Conversation + Tools │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │     Ollama Server     │
                         │ Qwen 2.5 3B Instruct  │
                         └───────────┬───────────┘
                                     │
                              tool/function call
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │     Tool Registry     │
                         │ Validation + Dispatch │
                         └───────────┬───────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    ▼                ▼                ▼
                read_file        write_file      list_directory
                    │                │                │
                    └────────────────┼────────────────┘
                                     │
                                     ▼
                              Tool execution
                                     │
                                     ▼
                              Tool result
                                     │
                                     ▼
                              Qwen 2.5 3B
                                     │
                                     ▼
                              Final response
```

### Responsibility boundaries

Echo deliberately separates model inference from host-side execution:

* **Qwen** decides what it needs to accomplish and which tool to request.
* **Echo** validates the requested tool call.
* **Echo** enforces workspace restrictions.
* **Echo** requests confirmation for sensitive operations.
* **Echo** executes the tool on the host.
* **Echo** returns the tool result to the model.
* **Qwen** produces the final response.

The model never receives unrestricted access to the host filesystem.

---

## Features

### Local AI inference

* Direct integration with the local Ollama server.
* Uses **Qwen 2.5 3B Instruct** by default.
* No cloud inference is required.
* Designed to run on relatively constrained hardware.

### Controlled filesystem tools

Echo currently provides host-side filesystem operations such as:

* `read_file`
* `write_file`
* `edit_file`
* `list_directory`
* directory tree inspection

All filesystem operations are restricted to the configured workspace.

For example:

```text
workspace/
├── resume_example.txt
├── test-echo.txt
└── test-location.txt
```

Attempts to access files outside the workspace are rejected by the host-side tool layer.

---

## Tool Validation and Safety

Tool calls are not executed directly from raw model output.

Before execution, Echo validates:

1. Tool name
2. JSON structure
3. Required parameters
4. Parameter types
5. Workspace boundaries
6. Filesystem path resolution
7. Operation-specific restrictions

Sensitive operations can require explicit user confirmation.

Example:

```text
[Tool Call]
write_file({
  "content": "Teste de confirmação.",
  "path": "teste.txt"
})

[Confirm] write_file
  Path: teste.txt
  Content: 21 characters
  Proceed? [y/N] y

[Tool Executed (OK) in 0.001s] -> Successfully wrote ...
```

This keeps the model's capabilities separate from the permissions granted to the host process.

---

## Loop Prevention

Small local models can occasionally repeat the same tool call or become stuck in a tool-use loop.

Echo includes safeguards to detect redundant tool calls and prevent runaway execution.

When the orchestrator detects repeated or invalid tool behavior, it can stop further tool execution and force the model toward a text response.

This provides an additional control layer between local inference and host-side execution.

---

## Interactive REPL

Running Echo without a prompt starts the interactive terminal interface:

```bash
python run_echo.py
```

Example:

```text
> Hello

Hi there! How can I assist you today?

[1 iteration | 0 tools | response 37 chars | total 2.95s | completed]
```

The REPL supports:

* Multiline input
* Persistent command history
* Slash commands
* Command completion
* Tool execution feedback
* Interactive tool confirmation
* Model switching
* Workspace switching
* Execution statistics

Multiline input can be entered directly in the prompt.

---

## Persistent History

Echo stores interactive input in:

```text
~/.echo/history.jsonl
```

The history uses JSONL rather than `prompt_toolkit`'s proprietary `FileHistory` format.

Each entry contains:

```json
{
  "text": "user input",
  "timestamp": "2026-09-23T20:32:29.894085+00:00",
  "workspace": "C:\\Users\\Sans\\echo\\workspace"
}
```

The history is used by the REPL for navigation with the **Up** and **Down** arrow keys.

History entries are also associated with the workspace in which they were created.

Echo avoids storing:

* Empty inputs
* Inputs beginning with a space
* Consecutive duplicate entries
* `/exit` and `/quit`

---

## Slash Commands

The interactive REPL provides several local commands.

| Command             | Description                             |
| ------------------- | --------------------------------------- |
| `/help`             | Show available commands and shortcuts   |
| `/clear`            | Clear the terminal screen               |
| `/model <name>`     | Switch the active model                 |
| `/models`           | List installed models and select one    |
| `/workspace <path>` | Change the active workspace             |
| `/stats`            | Show statistics from the last execution |
| `/tree`             | Show the workspace directory tree       |
| `/ls`               | List workspace contents                 |
| `/new`              | Start a new conversation                |
| `/exit`             | Exit Echo                               |
| `/quit`             | Alias for `/exit`                       |

Commands beginning with `/` are handled by the Echo CLI and are not sent to the language model.

---

## Terminal Interface

The interactive interface is built around `prompt_toolkit`.

The input layer provides:

* Command history
* Arrow-key history navigation
* Slash-command completion
* Multiline input
* Ctrl+C cancellation
* Ctrl+J newline insertion
* Paste handling
* Tool confirmation prompts

The CLI intentionally keeps the visual interface lightweight while providing enough interaction for day-to-day local agent usage.

---

## Configuration

Echo stores persistent configuration in:

```text
~/.echo/config.json
```

The configuration currently stores settings such as:

```json
{
  "model": "qwen2.5:3b-instruct",
  "workspace": "C:\\Users\\Sans\\echo\\workspace"
}
```

The workspace determines the root directory available to filesystem tools.

---

## Installation

### Requirements

* Python 3.10+
* Ollama
* A locally installed compatible model
* `requests`
* `prompt_toolkit`

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

If installing manually:

```bash
pip install requests prompt_toolkit
```

### Ollama

Install and run Ollama locally, then pull the default model:

```bash
ollama pull qwen2.5:3b-instruct
```

Verify that Ollama is available:

```bash
ollama list
```

The default Ollama endpoint is:

```text
http://localhost:11434
```

---

## Usage

### Interactive REPL

```bash
python run_echo.py
```

Then enter prompts directly:

```text
> List the files in the workspace
```

or:

```text
> Create a file called test.txt containing:
Test successful.
```

Echo will request the appropriate tool, validate it, ask for confirmation when required, execute it, and return the result to the model.

---

### Single Prompt

Echo can also be started with a prompt directly:

```bash
python run_echo.py "Create a file named test-echo.txt containing exactly: TOOL CALL OK"
```

---

### Health Check

Check the local Ollama connection:

```bash
python run_echo.py --check-health
```

---

### Custom Workspace

Specify a different workspace:

```bash
python run_echo.py --workspace /path/to/workspace "List all files"
```

On Windows:

```powershell
python run_echo.py --workspace "C:\path\to\workspace" "List all files"
```

The configured workspace becomes the root boundary for filesystem tools.

---

## Example

A typical tool-assisted interaction looks like this:

```text
> Crie um arquivo chamado teste.txt com o conteúdo:
Teste de confirmação.

[Tool Call]
write_file({
  "content": "Teste de confirmação.",
  "path": "teste.txt"
})

[Confirm] write_file
  Path: teste.txt
  Content: 21 characters
  Proceed? [y/N] y

[Tool Executed (OK) in 0.001s] -> Successfully wrote 23 bytes to 'teste.txt'. Location: .../workspace/teste.txt

O arquivo teste.txt foi criado com sucesso.
```

The important architectural detail is that the model does not directly write the file. It requests `write_file`; Echo validates the request and performs the actual filesystem operation.

---

## Testing

Run the unit tests:

```bash
python -m unittest tests/test_tools_unit.py
```

Run live end-to-end integration tests:

```bash
python -m unittest tests/test_live_experiment.py
```

The live tests require Ollama and the configured local model.

Run the complete test suite:

```bash
python -m unittest discover tests -v
```

---

## Project Structure

A simplified project structure:

```text
echo/
├── echo/
│   ├── cli.py
│   ├── orchestrator.py
│   ├── config.py
│   └── tools/
│       ├── filesystem.py
│       └── ...
├── tests/
│   ├── test_tools_unit.py
│   └── test_live_experiment.py
├── workspace/
├── run_echo.py
├── requirements.txt
└── README.md
```

The exact structure may evolve as additional tools and agent capabilities are introduced.

---

## Design Goals

Echo is an experimental component of a broader local-agent environment.

Its main design goals are:

* Keep local inference inexpensive.
* Keep host-side permissions explicit.
* Keep tool execution deterministic.
* Prevent models from directly accessing the host.
* Make tool calls observable.
* Require confirmation for potentially destructive operations.
* Maintain a persistent but lightweight REPL experience.
* Keep the implementation understandable and easy to modify.
* Provide a foundation for future multi-agent workflows.

---

## Current Scope

Echo is currently focused on local software-engineering support tasks, particularly:

* Filesystem inspection
* File creation
* File modification
* Workspace navigation
* Context preparation
* Repetitive development tasks
* Local agent experimentation

The project is intentionally small and experimental. Additional tools, agent capabilities, and integrations can be added without giving the underlying model unrestricted host access.

