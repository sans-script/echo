//! `run_command`: runs a shell command in the workspace. Every call needs the
//! user's confirmation (the registry marks it), runs without stdin and is
//! stopped when it exceeds its timeout.

use crate::tools::registry::{ToolDefinition, ToolRegistry};
use serde_json::{Value, json};
use std::{
    io::Read,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::Arc,
    thread,
    time::{Duration, Instant},
};

pub const DEFAULT_TIMEOUT_SECS: u64 = 60;
pub const MAX_TIMEOUT_SECS: u64 = 300;

/// Name of the shell commands run in, as shown to the user and the model.
pub fn shell_name() -> &'static str {
    if cfg!(windows) { "PowerShell" } else { "sh" }
}

fn shell_command(command: &str) -> Command {
    if cfg!(windows) {
        let mut shell = Command::new("powershell");
        // UTF-8 output so accents survive; no profile keeps startup fast.
        shell.args([
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            &format!("[Console]::OutputEncoding=[System.Text.Encoding]::UTF8; {command}"),
        ]);
        shell
    } else {
        let mut shell = Command::new("sh");
        shell.args(["-c", command]);
        shell
    }
}

pub fn timeout_from(arguments: &serde_json::Map<String, Value>) -> Duration {
    let seconds = arguments
        .get("timeout_seconds")
        .and_then(Value::as_u64)
        .unwrap_or(DEFAULT_TIMEOUT_SECS)
        .clamp(1, MAX_TIMEOUT_SECS);
    Duration::from_secs(seconds)
}

/// Runs `command` in `cwd`. A non-zero exit code or a timeout is an error
/// whose message still carries the output, so the model can react to it.
pub fn run_command(cwd: &Path, command: &str, timeout: Duration, max_chars: usize) -> Result<String, String> {
    if command.trim().is_empty() {
        return Err("command cannot be empty.".into());
    }
    let mut child = shell_command(command)
        .current_dir(cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("Could not start {}: {e}", shell_name()))?;

    // Drain both pipes on their own threads so a chatty command can't fill a
    // pipe buffer and block forever.
    let drain = |pipe: Option<Box<dyn Read + Send>>| {
        thread::spawn(move || {
            let mut bytes = Vec::new();
            if let Some(mut pipe) = pipe {
                let _ = pipe.read_to_end(&mut bytes);
            }
            String::from_utf8_lossy(&bytes).into_owned()
        })
    };
    let stdout = drain(child.stdout.take().map(|p| Box::new(p) as Box<dyn Read + Send>));
    let stderr = drain(child.stderr.take().map(|p| Box::new(p) as Box<dyn Read + Send>));

    let started = Instant::now();
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Some(status),
            Ok(None) if started.elapsed() >= timeout => {
                let _ = child.kill();
                let _ = child.wait();
                break None;
            }
            Ok(None) => thread::sleep(Duration::from_millis(50)),
            Err(error) => return Err(format!("Could not wait for the command: {error}")),
        }
    };
    let (stdout, stderr) = (stdout.join().unwrap_or_default(), stderr.join().unwrap_or_default());

    let mut output = stdout.trim_end().to_string();
    if !stderr.trim().is_empty() {
        output.push_str(&format!("\n[stderr]\n{}", stderr.trim_end()));
    }
    let output = cut_middle(&output, max_chars);

    match status {
        None => Err(format!(
            "Command timed out after {}s and was stopped. Output so far:\n{output}",
            timeout.as_secs()
        )),
        Some(status) if status.success() => Ok(format!(
            "Exit code: 0\n{}",
            if output.is_empty() { "(no output)" } else { &output }
        )),
        Some(status) => Err(format!(
            "Exit code: {}\n{output}",
            status.code().map_or("unknown".into(), |code| code.to_string())
        )),
    }
}

/// Keeps the start and the end of long output (compiler errors come first,
/// test summaries last) and drops the middle.
fn cut_middle(text: &str, max_chars: usize) -> String {
    if text.len() <= max_chars {
        return text.to_string();
    }
    let head = crate::orchestrator::floor_char_boundary(text, max_chars * 2 / 5);
    let mut tail = text.len() - max_chars * 3 / 5;
    while !text.is_char_boundary(tail) {
        tail += 1;
    }
    format!(
        "{}\n... [{} characters omitted] ...\n{}",
        &text[..head],
        tail - head,
        &text[tail..]
    )
}

pub fn register_shell_tools(registry: &mut ToolRegistry, workspace: &Path) -> Result<(), String> {
    let cwd: PathBuf = workspace.to_path_buf();
    // Leave room for the registry's own truncation note.
    let budget = registry.max_output_chars.saturating_sub(200).max(200);
    registry.register(ToolDefinition::new(
        "run_command",
        format!(
            "Run a non-interactive {} command in the workspace directory, e.g. to compile, run a program, run tests or use git. Returns the exit code and output. Stops after timeout_seconds (default {DEFAULT_TIMEOUT_SECS}, max {MAX_TIMEOUT_SECS}). The user approves every command.",
            shell_name()
        ),
        json!({"type":"object","properties":{"command":{"type":"string"},"timeout_seconds":{"type":"integer"}},"required":["command"],"additionalProperties":false}),
        true,
        Arc::new(move |a: &serde_json::Map<String, Value>| {
            run_command(&cwd, a["command"].as_str().unwrap_or(""), timeout_from(a), budget)
        }),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runs_in_the_workspace_and_reports_exit_codes() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("marker.txt"), "hi").unwrap();
        let listing = if cfg!(windows) { "Get-ChildItem -Name" } else { "ls" };

        let ok = run_command(dir.path(), listing, Duration::from_secs(30), 4000).unwrap();
        assert!(ok.starts_with("Exit code: 0") && ok.contains("marker.txt"), "{ok}");

        let failed = run_command(dir.path(), "exit 3", Duration::from_secs(30), 4000).unwrap_err();
        assert!(failed.starts_with("Exit code: 3"), "{failed}");
    }

    #[test]
    fn long_running_commands_time_out() {
        let dir = tempfile::tempdir().unwrap();
        let wait = if cfg!(windows) { "Start-Sleep -Seconds 20" } else { "sleep 20" };
        let started = Instant::now();
        let error = run_command(dir.path(), wait, Duration::from_secs(2), 4000).unwrap_err();
        assert!(error.contains("timed out after 2s"), "{error}");
        assert!(started.elapsed() < Duration::from_secs(10));
    }

    #[test]
    fn long_output_keeps_head_and_tail() {
        let text = format!("{}{}", "a".repeat(5000), "z".repeat(5000));
        let cut = cut_middle(&text, 1000);
        assert!(cut.starts_with("aaaa") && cut.ends_with("zzzz"));
        assert!(cut.contains("characters omitted"));
        assert!(cut.len() < 1100);
    }
}
