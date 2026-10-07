use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Clear, Paragraph, Wrap},
};

use crate::{
    app::{CONFIRM_QUESTION, ConfirmTone, EchoApp, MessageRole, Overlay, confirmation_segments},
    logo::LogoTone,
    markdown,
    logo_frames::{ECHO_LOGO, LOGO_HEIGHT},
    spinner::{self, Shade},
};

const DIVIDER: &str = "─";

const TEXT: Color = Color::Rgb(212, 212, 212);
const WHITE: Color = Color::Rgb(255, 255, 255);
const LIGHT_GRAY: Color = Color::Rgb(192, 192, 192);
const GRAY: Color = Color::Rgb(128, 128, 128);
const DIM_GRAY: Color = Color::Rgb(95, 95, 95);
const CYAN: Color = Color::Cyan;
const GREEN: Color = Color::Green;
const YELLOW: Color = Color::Yellow;
const RED: Color = Color::Red;
const SCROLL_BG: Color = Color::Rgb(48, 48, 48);
const SCROLL_THUMB: Color = Color::Rgb(144, 144, 144);

/// The composer grows with its content up to this many rows.
const MAX_COMPOSER_ROWS: u16 = 8;
/// Room for the summary plus the "+"/"-" preview of the change.
const MAX_CONFIRMATION_ROWS: u16 = 14;

pub fn draw(frame: &mut Frame, app: &mut EchoApp) {
    let area = frame.area();

    if app.overlay == Overlay::ModelPicker {
        draw_model_picker_layout(frame, app, area);
        return;
    }

    if app.confirmation.is_some() {
        draw_confirmation_layout(frame, app, area);
        return;
    }

    let input_height = if let Some(search) = &app.search {
        let width = usize::from(area.width.max(1));
        let text = search_line(&search.query, &search.text);
        (text.chars().count() + 1).div_ceil(width).clamp(1, 3) as u16
    } else {
        app.composer
            .wrapped_height(area.width)
            .clamp(1, MAX_COMPOSER_ROWS)
    };

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(LOGO_HEIGHT as u16),
            Constraint::Length(1),
            Constraint::Min(1),
            Constraint::Length(input_height),
            Constraint::Length(1),
            Constraint::Length(1),
        ])
        .split(area);

    draw_header(frame, chunks[0], app);
    draw_divider(frame, chunks[1]);

    draw_middle(frame, chunks[2], app);

    // Slash suggestions float over the bottom of the conversation, right
    // above the input. The cells under the menu are cleared first, so the
    // menu has the same (terminal) background as the chat and the chat text
    // never shows through it.
    let menu_height = completion_height(app);
    if menu_height > 0 && menu_height <= chunks[2].height {
        let area = chunks[2];
        let menu = Rect {
            y: area.y + area.height - menu_height,
            height: menu_height,
            ..area
        };
        frame.render_widget(Clear, menu);
        draw_completion(frame, menu, app);
    }

    if let Some(search) = &app.search {
        frame.render_widget(
            Paragraph::new(search_line(&search.query, &search.text))
                .style(Style::default().fg(WHITE))
                .wrap(Wrap { trim: false }),
            chunks[3],
        );
    } else {
        draw_composer(frame, chunks[3], app);
    }
    draw_divider(frame, chunks[4]);
    draw_footer(frame, chunks[5], app);
}

fn search_line(query: &str, found: &str) -> String {
    format!("(reverse-i-search)`{query}': {found}")
}

fn draw_confirmation_layout(frame: &mut Frame, app: &mut EchoApp, area: Rect) {
    let Some(confirmation) = &app.confirmation else {
        return;
    };
    let details = confirmation_lines(&confirmation.details);
    let details_height = (details.len() as u16).clamp(1, MAX_CONFIRMATION_ROWS);

    let mut answer = vec![Span::styled(CONFIRM_QUESTION, Style::default().fg(TEXT))];
    if let Some(choice) = confirmation.choice {
        let tone = if choice { ConfirmTone::Yes } else { ConfirmTone::No };
        answer.push(Span::styled(if choice { "y" } else { "n" }, tone_style(tone)));
    }
    if app.composer.cursor_visible() {
        answer.push(Span::styled("█", Style::default().fg(WHITE)));
    }

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(LOGO_HEIGHT as u16),
            Constraint::Length(1),
            Constraint::Min(1),
            Constraint::Length(details_height),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
        ])
        .split(area);

    draw_header(frame, chunks[0], app);
    draw_divider(frame, chunks[1]);
    draw_middle(frame, chunks[2], app);
    frame.render_widget(Paragraph::new(details), chunks[3]);
    frame.render_widget(Paragraph::new(Line::from(answer)), chunks[5]);
    draw_divider(frame, chunks[6]);
    draw_footer(frame, chunks[7], app);
}

fn draw_model_picker_layout(frame: &mut Frame, app: &mut EchoApp, area: Rect) {
    let list_height = app.model_picker_rows() as u16;
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(LOGO_HEIGHT as u16),
            Constraint::Length(1),
            Constraint::Min(1),
            Constraint::Length(list_height),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
        ])
        .split(area);

    draw_header(frame, chunks[0], app);
    draw_divider(frame, chunks[1]);
    draw_middle(frame, chunks[2], app);
    draw_model_list(frame, chunks[3], app);

    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(
            "Up/Down to move, Enter to select, Esc to cancel",
            Style::default().fg(GRAY),
        ))),
        chunks[5],
    );

    draw_divider(frame, chunks[6]);
    draw_footer(frame, chunks[7], app);
}

fn info_lines(app: &EchoApp) -> [String; 4] {
    [
        "Echo CLI 0.1.0".into(),
        "Local AI Support Assistant".into(),
        format!("{} (via Ollama @ {})", app.model, app.ollama_url),
        format!("Workspace: {}", app.workspace.display()),
    ]
}

/// Static banner as plain text, for the screen left behind on exit.
pub fn banner_lines(app: &EchoApp) -> Vec<String> {
    let info = info_lines(app);
    ECHO_LOGO
        .iter()
        .enumerate()
        .map(|(index, logo)| {
            format!(
                "  {logo}   {}",
                info.get(index).map(String::as_str).unwrap_or("")
            )
        })
        .collect()
}

fn draw_header(frame: &mut Frame, area: Rect, app: &EchoApp) {
    let logo = app.logo.current();
    let logo_style = Style::default().fg(match logo.tone {
        LogoTone::White => WHITE,
        LogoTone::Gray => GRAY,
    });
    let info = info_lines(app);

    let lines = logo
        .lines
        .iter()
        .enumerate()
        .map(|(index, logo_line)| {
            let mut row = Line::from(vec![
                Span::raw("  "),
                Span::styled(logo_line.clone(), logo_style),
                Span::raw("   "),
            ]);
            if let Some(text) = info.get(index) {
                let style = if index == 0 {
                    Style::default().fg(WHITE).add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(GRAY)
                };
                row.push_span(Span::styled(text.clone(), style));
            }
            row
        })
        .collect::<Vec<_>>();

    frame.render_widget(Paragraph::new(lines), area);
}

fn draw_middle(frame: &mut Frame, area: Rect, app: &mut EchoApp) {
    if area.height == 0 || area.width == 0 {
        return;
    }

    let lines = build_middle_lines(app);
    let viewport_height = area.height as usize;

    // Python's scrollbar does a two-pass measurement. The first pass decides
    // whether the scrollbar is needed; only then do we reserve one column and
    // reflow the text at the narrower width.
    let full_content_height = visual_height(&lines, area.width as usize);
    let overflow = full_content_height > viewport_height;

    let text_width = if overflow {
        area.width.saturating_sub(1) as usize
    } else {
        area.width as usize
    };

    let content_height = visual_height(&lines, text_width);
    let max_scroll = content_height.saturating_sub(viewport_height);

    app.scroll.max_offset = max_scroll;
    app.scroll.viewport_height = viewport_height;

    if app.scroll.follow_end {
        app.scroll.offset = max_scroll;
    } else {
        app.scroll.offset = app.scroll.offset.min(max_scroll);
    }

    let offset = app.scroll.offset;

    let text_area = if overflow {
        Rect {
            width: area.width.saturating_sub(1),
            ..area
        }
    } else {
        area
    };

    frame.render_widget(
        Paragraph::new(lines)
            .scroll((offset.min(u16::MAX as usize) as u16, 0))
            .wrap(Wrap { trim: false }),
        text_area,
    );

    if overflow {
        draw_scrollbar(frame, area, content_height, viewport_height, offset);
    }
}

fn draw_scrollbar(
    frame: &mut Frame,
    area: Rect,
    content_height: usize,
    viewport_height: usize,
    offset: usize,
) {
    if area.width == 0 || viewport_height == 0 || content_height <= viewport_height {
        return;
    }

    let track_height = viewport_height;
    let thumb_height = (track_height * viewport_height / content_height)
        .max(1)
        .min(track_height);
    let scroll_range = content_height.saturating_sub(viewport_height);
    let max_thumb_top = track_height.saturating_sub(thumb_height);

    let thumb_top = if scroll_range == 0 {
        0
    } else {
        (max_thumb_top * offset.min(scroll_range) + scroll_range / 2) / scroll_range
    };

    let x = area.x + area.width - 1;
    let buffer = frame.buffer_mut();

    for row in 0..track_height {
        let y = area.y + row as u16;
        let in_thumb = (row as usize) >= thumb_top && (row as usize) < thumb_top + thumb_height;
        let style = if in_thumb {
            Style::default().fg(SCROLL_THUMB).bg(SCROLL_THUMB)
        } else {
            Style::default().fg(SCROLL_BG).bg(SCROLL_BG)
        };

        buffer[(x, y)].set_symbol(" ").set_style(style);
    }
}

fn build_middle_lines(app: &EchoApp) -> Vec<Line<'static>> {
    match app.overlay {
        Overlay::Help => {
            let mut lines = help_lines();
            lines.push(Line::default());
            lines.extend(chat_lines(app));
            lines
        }
        Overlay::None => chat_lines(app),
        Overlay::ModelPicker => chat_lines(app),
    }
}

fn chat_lines(app: &EchoApp) -> Vec<Line<'static>> {
    let mut lines = Vec::new();

    for message in &app.messages {
        if message.role == MessageRole::Help {
            lines.extend(help_lines());
            lines.push(Line::default());
            continue;
        }
        if message.role == MessageRole::Confirmation {
            lines.extend(confirmation_lines(&message.content));
            lines.push(Line::default());
            continue;
        }
        if message.role == MessageRole::Assistant {
            // The model writes markdown; show it styled instead of raw.
            for (index, spans) in markdown::render(&message.content).into_iter().enumerate() {
                let prefix = if index == 0 {
                    Span::styled("> ", Style::default().fg(DIM_GRAY))
                } else {
                    Span::raw("  ")
                };
                lines.push(Line::from([vec![prefix], spans].concat()));
            }
            lines.push(Line::default());
            continue;
        }

        let prefixed = message.role == MessageRole::User;
        let (first, rest) = match message.role {
            MessageRole::User => {
                let style = Style::default().fg(DIM_GRAY).add_modifier(Modifier::BOLD);
                (style, style)
            }
            MessageRole::Tool | MessageRole::Muted => {
                (Style::default().fg(GRAY), Style::default().fg(GRAY))
            }
            MessageRole::ToolResult => (Style::default().fg(GRAY), Style::default().fg(WHITE)),
            MessageRole::Plain => (Style::default().fg(TEXT), Style::default().fg(TEXT)),
            MessageRole::Success => (Style::default().fg(GREEN), Style::default().fg(GREEN)),
            MessageRole::Usage => (Style::default().fg(YELLOW), Style::default().fg(GRAY)),
            MessageRole::Error => (Style::default().fg(RED), Style::default().fg(GRAY)),
            MessageRole::Assistant | MessageRole::Help | MessageRole::Confirmation => {
                unreachable!()
            }
        };

        for (index, line) in message.content.split('\n').enumerate() {
            let mut spans = Vec::with_capacity(2);
            if prefixed {
                spans.push(if index == 0 {
                    Span::styled("> ", Style::default().fg(DIM_GRAY))
                } else {
                    Span::raw("  ")
                });
            }
            let style = if index == 0 { first } else { rest };
            spans.push(Span::styled(line.to_owned(), style));
            lines.push(Line::from(spans));
        }

        lines.push(Line::default());
    }

    if let Some(since) = app.waiting_since {
        lines.push(loading_line(spinner::frame_index(since)));
    }

    lines
}

/// A confirmation request with its header, path, counts and the "+"/"-"
/// preview of the change in color.
fn confirmation_lines(text: &str) -> Vec<Line<'static>> {
    text.lines()
        .enumerate()
        .map(|(index, line)| {
            Line::from(
                confirmation_segments(index, line)
                    .into_iter()
                    .map(|(tone, segment)| Span::styled(segment, tone_style(tone)))
                    .collect::<Vec<_>>(),
            )
        })
        .collect()
}

fn tone_style(tone: ConfirmTone) -> Style {
    let bold = |color| Style::default().fg(color).add_modifier(Modifier::BOLD);
    match tone {
        ConfirmTone::Header => bold(WHITE),
        ConfirmTone::Path => bold(CYAN),
        ConfirmTone::Created | ConfirmTone::Yes => bold(GREEN),
        ConfirmTone::Modified => bold(YELLOW),
        ConfirmTone::Deleted | ConfirmTone::No => bold(RED),
        ConfirmTone::Added => Style::default().fg(GREEN),
        ConfirmTone::Removed => Style::default().fg(RED),
        ConfirmTone::Muted => Style::default().fg(GRAY),
        ConfirmTone::Text => Style::default().fg(TEXT),
    }
}

fn loading_line(frame: usize) -> Line<'static> {
    let (glyph, shaded) = spinner::loading_frame(frame);
    let mut spans = vec![Span::styled(
        format!("{glyph} "),
        Style::default().fg(GRAY),
    )];
    spans.extend(shaded.into_iter().map(|(ch, shade)| {
        let style = match shade {
            Shade::Bright => Style::default().fg(WHITE).add_modifier(Modifier::BOLD),
            Shade::White => Style::default().fg(WHITE),
            Shade::Light => Style::default().fg(LIGHT_GRAY),
            Shade::Gray => Style::default().fg(GRAY),
        };
        Span::styled(ch.to_string(), style)
    }));
    Line::from(spans)
}

fn help_lines() -> Vec<Line<'static>> {
    let mut lines = vec![
        Line::from(Span::styled(
            "Echo Commands",
            Style::default().fg(WHITE).add_modifier(Modifier::BOLD),
        )),
        Line::default(),
    ];

    for (command, description) in crate::app::SLASH_COMMANDS {
        lines.push(Line::from(vec![
            Span::raw("  "),
            Span::styled(format!("{command:<12}"), Style::default().fg(CYAN)),
            Span::styled(format!(" {description}"), Style::default().fg(GRAY)),
        ]));
    }

    lines.push(Line::default());
    lines.push(Line::from(Span::styled(
        "Shortcuts",
        Style::default().fg(WHITE).add_modifier(Modifier::BOLD),
    )));
    lines.push(Line::default());

    for (key, description) in [
        ("Enter", "Submit prompt"),
        ("Ctrl+J", "Insert newline (multi-line)"),
        ("Ctrl+C", "Clear input / interrupt"),
        ("Ctrl+D", "Exit"),
        ("Esc", "Cancel the running request"),
        ("↑ / ↓", "Navigate history"),
        ("Ctrl+R", "Reverse history search"),
        ("/", "Type to see command autocomplete"),
    ] {
        lines.push(Line::from(vec![
            Span::raw("  "),
            Span::styled(format!("{key:<11}"), Style::default().fg(CYAN)),
            Span::styled(format!(" {description}"), Style::default().fg(GRAY)),
        ]));
    }

    lines
}

const MAX_COMPLETION_ROWS: usize = 8;

/// Rows used by the slash suggestions: the visible commands plus one blank
/// row above and one below. Zero when no suggestions are shown.
fn completion_height(app: &EchoApp) -> u16 {
    if !app.completion_active() {
        return 0;
    }
    app.matching_commands().len().min(MAX_COMPLETION_ROWS) as u16 + 2
}

fn draw_completion(frame: &mut Frame, menu_area: Rect, app: &EchoApp) {
    let matches = app.matching_commands();
    if matches.is_empty() || menu_area.height == 0 || menu_area.width == 0 {
        return;
    }

    let selected = app.completion.selected.min(matches.len() - 1);
    let start = selected.saturating_sub(MAX_COMPLETION_ROWS - 1);
    let visible = &matches[start..matches.len().min(start + MAX_COMPLETION_ROWS)];
    let menu_height = visible.len() as u16 + 2;

    let name_width = visible
        .iter()
        .map(|(name, _)| name.len())
        .max()
        .unwrap_or(0);

    let mut rows = Vec::with_capacity(menu_height as usize);
    rows.push(Line::default());

    for (offset, (command, description)) in visible.iter().enumerate() {
        let index = start + offset;
        let is_selected = index == selected;
        let marker = if is_selected { "> " } else { "  " };
        let name = format!("{:<width$}", command, width = name_width);

        let name_style = if is_selected {
            Style::default().fg(WHITE).add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(TEXT)
        };
        let meta_style = if is_selected {
            Style::default().fg(Color::Rgb(204, 204, 204))
        } else {
            Style::default().fg(GRAY)
        };

        rows.push(Line::from(vec![
            Span::styled(format!("{}{}", marker, name), name_style),
            Span::styled(format!("   {}", description), meta_style),
        ]));
    }

    rows.push(Line::default());

    frame.render_widget(Paragraph::new(rows).wrap(Wrap { trim: false }), menu_area);
}

fn draw_model_list(frame: &mut Frame, area: Rect, app: &EchoApp) {
    let picker = &app.model_picker;
    let rows = app.model_picker_rows();
    let start = picker.selected.saturating_sub(rows.saturating_sub(1));
    let current = crate::ollama::normalize_model_name(&app.model);

    let lines = picker
        .models
        .iter()
        .enumerate()
        .skip(start)
        .take(rows)
        .map(|(index, model)| {
            let selected = index == picker.selected;
            let marker = if selected { "> " } else { "  " };
            let style = if selected {
                Style::default().fg(WHITE).add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(TEXT)
            };
            let suffix = if crate::ollama::normalize_model_name(&model.name) == current {
                "  (current)"
            } else {
                ""
            };

            Line::from(Span::styled(format!("{marker}{}{suffix}", model.name), style))
        })
        .collect::<Vec<_>>();

    frame.render_widget(Paragraph::new(lines), area);
}

fn draw_composer(frame: &mut Frame, area: Rect, app: &EchoApp) {
    // Keep the end of a tall multi-line draft (where the cursor usually is)
    // visible once it exceeds the maximum composer height.
    let rows = app.composer.wrapped_height(area.width);
    let scroll = rows.saturating_sub(area.height);
    frame.render_widget(
        Paragraph::new(app.composer.visible_with_cursor())
            .style(Style::default().fg(WHITE))
            .wrap(Wrap { trim: false })
            .scroll((scroll, 0)),
        area,
    );
}

fn draw_footer(frame: &mut Frame, area: Rect, app: &EchoApp) {
    let left = "? for shortcuts";
    let right = &app.model;
    let padding = (area.width as usize)
        .saturating_sub(left.len() + right.chars().count())
        .max(1);

    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(left, Style::default().fg(DIM_GRAY)),
            Span::raw(" ".repeat(padding)),
            Span::styled(right.clone(), Style::default().fg(DIM_GRAY)),
        ])),
        area,
    );
}

fn draw_divider(frame: &mut Frame, area: Rect) {
    frame.render_widget(
        Paragraph::new(DIVIDER.repeat(area.width as usize)).style(Style::default().fg(DIM_GRAY)),
        area,
    );
}

fn visual_height(lines: &[Line<'static>], width: usize) -> usize {
    if lines.is_empty() {
        return 0;
    }

    let width = width.max(1);

    lines
        .iter()
        .map(|line| line.width().max(1).div_ceil(width))
        .sum()
}

#[cfg(test)]
mod tests {
    use super::{build_middle_lines, chat_lines, visual_height};
    use crate::app::{EchoApp, Message, MessageRole, Overlay};
    use ratatui::{Terminal, backend::TestBackend, text::Line};
    use std::time::Instant;

    #[test]
    fn middle_content_is_empty_without_messages_or_overlay() {
        let app = EchoApp::new();
        let lines = build_middle_lines(&app);
        assert!(lines.is_empty());
        assert_eq!(visual_height(&lines, 120), 0);
    }

    #[test]
    fn help_content_precedes_chat_content() {
        let mut app = EchoApp::new();
        app.overlay = Overlay::Help;
        let lines = build_middle_lines(&app);
        assert!(lines.first().is_some());
        assert_eq!(lines.first().unwrap().width(), 13);
    }

    #[test]
    fn visual_height_accounts_for_wrapping() {
        let app = EchoApp::new();
        let mut lines = build_middle_lines(&app);
        lines.push(Line::from("x".repeat(121)));
        assert_eq!(visual_height(&lines, 120), 2);
    }

    #[test]
    fn only_user_and_assistant_lines_get_a_prefix() {
        let mut app = EchoApp::new();
        app.messages.push(Message {
            role: MessageRole::Assistant,
            content: "hi".into(),
        });
        app.messages.push(Message {
            role: MessageRole::Muted,
            content: "[stats]".into(),
        });
        let lines = chat_lines(&app);
        assert_eq!(lines[0].to_string(), "> hi");
        assert_eq!(lines[2].to_string(), "[stats]");
    }

    #[test]
    fn slash_suggestions_cover_the_chat_without_mixing_with_it() {
        let mut app = EchoApp::new();
        for n in 0..40 {
            app.messages.push(Message {
                role: MessageRole::Assistant,
                content: format!("chat line {n} with enough text to reach the menu columns"),
            });
        }
        app.composer.insert('/');
        app.completion.visible = true;

        let mut terminal = Terminal::new(TestBackend::new(80, 40)).unwrap();
        terminal.draw(|frame| super::draw(frame, &mut app)).unwrap();
        let buffer = terminal.backend().buffer();
        let rows = (0..buffer.area.height)
            .map(|y| {
                (0..buffer.area.width)
                    .map(|x| buffer[(x, y)].symbol())
                    .collect::<String>()
            })
            .collect::<Vec<_>>();

        // Input is 3 rows from the bottom (input, divider, footer).
        let input = rows.len() - 3;
        let commands = app.matching_commands().len().min(8);
        let first = input - 1 - commands;
        assert!(rows[input - 1].trim().is_empty(), "gap above the input");
        assert!(rows[first - 1].trim().is_empty(), "gap above the menu");
        assert!(rows[first].trim_start().starts_with("> /help"));
        for row in &rows[first - 1..input] {
            assert!(!row.contains("chat line"), "chat mixed into menu: {row}");
        }
        // The menu floats over the chat: the conversation is not pushed up,
        // so its newest line is hidden under the menu, and older lines
        // continue right above it.
        assert!(!rows.iter().any(|row| row.contains("chat line 39")));
        assert!(
            rows[first - 3..first - 1].iter().any(|row| row.contains("chat line")),
            "chat continues above the menu"
        );
    }

    #[test]
    fn spinner_is_the_last_line_while_waiting() {
        let mut app = EchoApp::new();
        app.waiting_since = Some(Instant::now());
        let lines = chat_lines(&app);
        assert_eq!(lines.last().unwrap().to_string(), "⠋ Loading...");
    }
}
