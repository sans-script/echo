mod app;
mod config;
mod highlight;
mod history;
mod input;
mod logo;
mod logo_frames;
mod markdown;
mod ollama;
mod oneshot;
mod orchestrator;
mod spinner;
mod tools;
mod ui;
mod workspace_helpers;

use std::{io, path::PathBuf, process::ExitCode, time::Duration};

use crossterm::{
    cursor::MoveTo,
    event::{DisableBracketedPaste, EnableBracketedPaste},
    execute,
    terminal::{
        self, Clear, ClearType, EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode,
        enable_raw_mode,
    },
};
use ratatui::{Terminal, backend::CrosstermBackend};

use app::EchoApp;
use config::EchoConfig;

const USAGE: &str = "\
Echo — Local AI Support Assistant with Tool Execution

Usage: echo [OPTIONS] [PROMPT]

Arguments:
  [PROMPT]  User prompt to execute. If omitted, runs in interactive mode.

Options:
  -w, --workspace <PATH>  Workspace directory path
  -m, --model <NAME>      Ollama model name
  -u, --url <URL>         Ollama server URL
  -v, --verbose           Print verbose execution tracing (default)
  -q, --quiet             Disable verbose output and the logo animation
      --check-health      Check Ollama connection and model availability, then exit
  -h, --help              Show this help";

#[derive(Debug, Default, PartialEq)]
struct Args {
    prompt: Option<String>,
    workspace: Option<String>,
    model: Option<String>,
    url: Option<String>,
    quiet: bool,
    check_health: bool,
}

fn parse_args(args: impl IntoIterator<Item = String>) -> Result<Args, String> {
    let mut parsed = Args::default();
    let mut prompt = Vec::new();
    let mut args = args.into_iter();

    while let Some(arg) = args.next() {
        let (flag, inline) = match arg.split_once('=') {
            Some((flag, value)) if arg.starts_with("--") => (flag.to_string(), Some(value.to_string())),
            _ => (arg.clone(), None),
        };
        let mut value = |name: &str| {
            inline
                .clone()
                .or_else(|| args.next())
                .ok_or_else(|| format!("argument {name}: expected one argument"))
        };
        match flag.as_str() {
            "-w" | "--workspace" => parsed.workspace = Some(value("--workspace")?),
            "-m" | "--model" => parsed.model = Some(value("--model")?),
            "-u" | "--url" => parsed.url = Some(value("--url")?),
            "-v" | "--verbose" => {}
            "-q" | "--quiet" => parsed.quiet = true,
            "--check-health" => parsed.check_health = true,
            "-h" | "--help" => return Err(String::new()),
            other if other.starts_with('-') && other.len() > 1 => {
                return Err(format!("unrecognized argument: {other}"));
            }
            _ => prompt.push(arg),
        }
    }

    if !prompt.is_empty() {
        parsed.prompt = Some(prompt.join(" "));
    }
    Ok(parsed)
}

/// Saved settings, overridden by any flag passed on the command line.
fn resolve_config(args: &Args) -> io::Result<EchoConfig> {
    let mut config = EchoConfig::load();
    if let Some(model) = &args.model {
        config.model = model.clone();
    }
    if let Some(url) = &args.url {
        config.ollama_url = url.trim_end_matches('/').to_string();
    }
    if let Some(workspace) = &args.workspace {
        let path = PathBuf::from(workspace);
        std::fs::create_dir_all(&path)?;
        config.workspace = workspace_helpers::display_path(std::fs::canonicalize(path)?);
    }
    Ok(config)
}

fn main() -> ExitCode {
    let args = match parse_args(std::env::args().skip(1)) {
        Ok(args) => args,
        Err(error) if error.is_empty() => {
            println!("{USAGE}");
            return ExitCode::SUCCESS;
        }
        Err(error) => {
            eprintln!("{USAGE}\n\nerror: {error}");
            return ExitCode::from(2);
        }
    };

    let config = match resolve_config(&args) {
        Ok(config) => config,
        Err(error) => {
            eprintln!("error: invalid workspace: {error}");
            return ExitCode::from(2);
        }
    };

    if args.check_health {
        return check_health(&config);
    }

    let verbose = !args.quiet;
    if let Some(prompt) = &args.prompt {
        return oneshot::run(config, prompt, verbose);
    }

    match run_interactive(config, verbose) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

fn check_health(config: &EchoConfig) -> ExitCode {
    println!("Checking Ollama connection at {}...", config.ollama_url);
    let result = ollama::OllamaClient::new(&config.ollama_url, Duration::from_secs(10))
        .and_then(|client| client.check_health(Some(&config.model)));
    match result {
        Ok(()) => {
            println!("[HEALTH OK] Ollama is reachable and model is available.");
            ExitCode::SUCCESS
        }
        Err(ollama::OllamaClientError::Malformed(message)) => {
            println!("[HEALTH ERROR] {message}");
            ExitCode::FAILURE
        }
        Err(error) => {
            println!("[HEALTH ERROR] {error}");
            ExitCode::FAILURE
        }
    }
}

fn run_interactive(config: EchoConfig, verbose: bool) -> io::Result<()> {
    let mut app = EchoApp::with_config(config, verbose);

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    // The mouse is deliberately not captured, so the terminal's own text
    // selection and copy keep working (as in the Python version). The
    // transcript scrolls with PageUp/PageDown and Ctrl+Home/Ctrl+End.
    execute!(stdout, EnterAlternateScreen)?;
    // Not available on every Windows console; pastes then arrive as key
    // bursts, which the app detects on its own.
    let _ = execute!(stdout, EnableBracketedPaste);

    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    terminal.hide_cursor()?;

    let result = app.run(&mut terminal);

    disable_raw_mode()?;
    let _ = execute!(terminal.backend_mut(), DisableBracketedPaste);
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    // Leave only the banner, a divider and "Exiting..." on screen.
    let width = terminal::size().map(|(width, _)| width).unwrap_or(80);
    execute!(io::stdout(), Clear(ClearType::All), MoveTo(0, 0))?;
    println!("{}", app.exit_screen(width));

    result
}

#[cfg(test)]
mod tests {
    use super::{Args, parse_args};

    fn parse(args: &[&str]) -> Result<Args, String> {
        parse_args(args.iter().map(|arg| arg.to_string()))
    }

    #[test]
    fn parses_python_compatible_flags() {
        let args = parse(&["-w", "ws", "--model=qwen", "-q", "List all files"]).unwrap();
        assert_eq!(args.workspace.as_deref(), Some("ws"));
        assert_eq!(args.model.as_deref(), Some("qwen"));
        assert!(args.quiet);
        assert_eq!(args.prompt.as_deref(), Some("List all files"));
    }

    #[test]
    fn rejects_unknown_flags_and_missing_values() {
        assert!(parse(&["--nope"]).is_err());
        assert!(parse(&["--model"]).is_err());
        assert_eq!(parse(&["--help"]), Err(String::new()));
        assert!(parse(&["--check-health"]).unwrap().check_health);
    }
}
