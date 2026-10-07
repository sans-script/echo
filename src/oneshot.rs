//! Non-interactive mode: `echo "prompt"` runs one turn, streams the answer to
//! stdout and exits with 0 when the turn completed.

use crate::{
    app::{
        CONFIRM_QUESTION, ConfirmTone, confirmation_details, confirmation_segments,
        pretty_arguments, tool_result_message,
    },
    config::EchoConfig,
    orchestrator::{EchoOrchestrator, OrchestratorEvent, RunStopReason, format_stats},
    spinner::{self, Shade},
    tools::registry::ToolRegistry,
};
use crossterm::style::Stylize;
use std::{
    cell::RefCell,
    io::{self, BufRead, IsTerminal, Write},
    process::ExitCode,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread::{self, JoinHandle},
    time::Instant,
};

pub fn run(config: EchoConfig, prompt: &str, verbose: bool) -> ExitCode {
    // Plain output when redirected: no colors and no spinner animation.
    let tty = io::stdout().is_terminal();
    let paint = Paint { tty };

    let workspace = config.workspace.clone();
    let mut tools = ToolRegistry::new(config.max_tool_output_chars);
    if let Err(error) = crate::tools::register_all(&mut tools, &workspace) {
        eprintln!("{} {error}", paint.error("[Error]"));
        return ExitCode::FAILURE;
    }
    let mut orchestrator = match EchoOrchestrator::new(config, tools) {
        Ok(orchestrator) => orchestrator,
        Err(error) => {
            eprintln!("{} {error}", paint.error("[Error]"));
            return ExitCode::FAILURE;
        }
    };
    orchestrator.tool_output_visible = verbose;

    if verbose {
        println!("Workspace: {}", workspace.display());
        println!("Model:     {}", orchestrator.config.model);
        println!("Prompt:    {prompt}");
        println!();
    }

    // Both callbacks stop the spinner; they never run at the same time.
    let spinner = RefCell::new(Spinner::default());
    let mut printing = false;
    let mut streamed = false;
    let result = orchestrator.run(
        prompt,
        None,
        |event| match event {
            OrchestratorEvent::IterationStarted if tty => spinner.borrow_mut().start(),
            OrchestratorEvent::IterationStarted => {}
            OrchestratorEvent::ContentDelta(delta) => {
                spinner.borrow_mut().stop();
                if !printing {
                    print!("{}", paint.gray("> "));
                    printing = true;
                }
                streamed = true;
                print!("{}", paint.white(&delta));
                let _ = io::stdout().flush();
            }
            OrchestratorEvent::ToolCallReceived {
                tool_name,
                arguments,
            } => {
                spinner.borrow_mut().stop();
                if printing {
                    println!();
                    printing = false;
                }
                if verbose {
                    println!("{}", paint.gray("[Tool Call]"));
                    let call = format!("{tool_name}({})", pretty_arguments(&arguments));
                    println!("{}", paint.gray(&call));
                }
            }
            OrchestratorEvent::ToolExecuted(record) => {
                if verbose {
                    let (_, content) = tool_result_message(&record);
                    println!("{}\n", paint.gray(&content));
                }
            }
            OrchestratorEvent::EmptyResponseRetry => {}
        },
        |tool_name, arguments| {
            spinner.borrow_mut().stop();
            println!();
            let details = confirmation_details(tool_name, arguments, &workspace);
            for (index, line) in details.lines().enumerate() {
                let colored = confirmation_segments(index, line)
                    .into_iter()
                    .map(|(tone, segment)| paint.tone(tone, &segment))
                    .collect::<String>();
                println!("{colored}");
            }
            print!("{CONFIRM_QUESTION}");
            let _ = io::stdout().flush();
            let mut answer = String::new();
            if io::stdin().lock().read_line(&mut answer).is_err() {
                return false;
            }
            confirmed(&answer)
        },
    );
    spinner.borrow_mut().stop();

    match (&result.stopped_reason, &result.error_message) {
        (RunStopReason::Error, Some(error)) => println!("\n{} {error}", paint.error("[Error]")),
        _ if !streamed => println!("{}{}", paint.gray("> "), paint.white(&result.final_response)),
        _ => println!(),
    }
    if verbose {
        let stats = format!("[{}]", format_stats(&result));
        println!("\n{}", paint.gray(&stats));
    }

    if result.stopped_reason == RunStopReason::Completed {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

/// An empty answer means yes, as in apt's [Y/n]. PowerShell 5.1 prefixes
/// piped input with a UTF-8 byte order mark, which is ignored.
fn confirmed(answer: &str) -> bool {
    let answer = answer.trim().trim_start_matches('\u{feff}').to_lowercase();
    matches!(answer.as_str(), "" | "y" | "yes")
}

/// Applies terminal colors only when stdout is a terminal.
#[derive(Clone, Copy)]
struct Paint {
    tty: bool,
}

impl Paint {
    fn gray(self, text: &str) -> String {
        if self.tty { text.dark_grey().to_string() } else { text.to_string() }
    }

    fn white(self, text: &str) -> String {
        if self.tty { text.white().to_string() } else { text.to_string() }
    }

    fn error(self, text: &str) -> String {
        if self.tty { text.red().bold().to_string() } else { text.to_string() }
    }

    fn tone(self, tone: ConfirmTone, text: &str) -> String {
        if !self.tty {
            return text.to_string();
        }
        let styled = match tone {
            ConfirmTone::Header => text.white().bold(),
            ConfirmTone::Path => text.cyan().bold(),
            ConfirmTone::Created | ConfirmTone::Yes => text.green().bold(),
            ConfirmTone::Modified => text.yellow().bold(),
            ConfirmTone::Deleted | ConfirmTone::No => text.red().bold(),
            ConfirmTone::Added => text.green(),
            ConfirmTone::Removed => text.red(),
            ConfirmTone::Muted => text.dark_grey(),
            ConfirmTone::Text => text.reset(),
        };
        styled.to_string()
    }
}

/// "⠋ Loading..." on the current line while waiting for the model.
#[derive(Default)]
struct Spinner {
    running: Option<(Arc<AtomicBool>, JoinHandle<()>)>,
}

impl Spinner {
    fn start(&mut self) {
        if self.running.is_some() {
            return;
        }
        let stop = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&stop);
        let handle = thread::spawn(move || {
            let since = Instant::now();
            while !flag.load(Ordering::Relaxed) {
                let (glyph, shaded) = spinner::loading_frame(spinner::frame_index(since));
                let mut line = format!("\r{} ", glyph.dark_grey());
                for (ch, shade) in shaded {
                    let styled = match shade {
                        Shade::Bright => ch.white().bold(),
                        Shade::White => ch.white(),
                        Shade::Light => ch.grey(),
                        Shade::Gray => ch.dark_grey(),
                    };
                    line.push_str(&styled.to_string());
                }
                print!("{line}");
                let _ = io::stdout().flush();
                thread::sleep(spinner::FRAME_TIME / 2);
            }
            // Erase the spinner line.
            print!("\r{}\r", " ".repeat(spinner::LOADING_TEXT.len() + 4));
            let _ = io::stdout().flush();
        });
        self.running = Some((stop, handle));
    }

    fn stop(&mut self) {
        if let Some((stop, handle)) = self.running.take() {
            stop.store(true, Ordering::Relaxed);
            let _ = handle.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::confirmed;

    #[test]
    fn confirmation_answers() {
        assert!(confirmed("\n"));
        assert!(confirmed("Y\r\n"));
        assert!(confirmed("\u{feff}y\r\n"));
        assert!(!confirmed("n\n"));
        assert!(!confirmed("nope\n"));
    }
}
