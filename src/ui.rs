use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Paragraph, Wrap},
};

use crate::app::{EchoApp, MessageRole, Overlay};

const DIVIDER: &str = "─";

const TEXT: Color = Color::Rgb(212, 212, 212);
const WHITE: Color = Color::Rgb(255, 255, 255);
const GRAY: Color = Color::Rgb(128, 128, 128);
const DIM_GRAY: Color = Color::Rgb(95, 95, 95);
const CYAN: Color = Color::Cyan;
const SCROLL_BG: Color = Color::Rgb(48, 48, 48);
const SCROLL_THUMB: Color = Color::Rgb(144, 144, 144);

const LOGO: [&str; 8] = [
    "         ::::::     ",
    "      :::     ::    ",
    "     :::     :::    ",
    "    ::::::::::      ",
    "    :::             ",
    "    :::        :    ",
    "    :::      :::    ",
    "      :::::::       ",
];

pub fn draw(frame: &mut Frame, app: &mut EchoApp) {
    let area = frame.area();

    if app.overlay == Overlay::ModelPicker {
        draw_model_picker_layout(frame, app, area);
        return;
    }

    let completion_height = if app.completion_active() {
        let count = app.matching_commands().len().min(8);
        if count > 0 { 1 + count as u16 } else { 0 }
    } else {
        0
    };

    let composer_height = app
        .composer
        .wrapped_height(area.width)
        .clamp(1, area.height.max(1));

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(8),
            Constraint::Length(1),
            Constraint::Min(1),
            Constraint::Length(completion_height),
            Constraint::Length(composer_height),
            Constraint::Length(1),
            Constraint::Length(1),
        ])
        .split(area);

    draw_header(frame, chunks[0], app);
    draw_divider(frame, chunks[1]);
    draw_middle(frame, chunks[2], app);

    if completion_height > 0 {
        draw_completion(frame, chunks[3], app);
    }

    draw_composer(frame, chunks[4], app);
    draw_divider(frame, chunks[5]);
    draw_footer(frame, chunks[6], app);
}

fn draw_model_picker_layout(frame: &mut Frame, app: &mut EchoApp, area: Rect) {
    let list_height = 2u16;
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(8),
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

fn draw_header(frame: &mut Frame, area: Rect, app: &EchoApp) {
    let logo_width = LOGO.iter().map(|line| line.len()).max().unwrap_or(0);
    let text_x = logo_width + 5;
    let mut lines = Vec::with_capacity(LOGO.len());

    for (index, logo) in LOGO.iter().enumerate() {
        let mut row = Line::from(Span::styled(*logo, Style::default().fg(WHITE)));
        row.push_span(Span::raw(" ".repeat(text_x.saturating_sub(logo.len()))));

        match index {
            0 => row.push_span(Span::styled("Echo CLI 0.1.0", Style::default().fg(WHITE))),
            1 => row.push_span(Span::styled(
                "Local AI Support Assistant",
                Style::default().fg(TEXT),
            )),
            2 => row.push_span(Span::styled(
                format!("{} (via Ollama @ http://localhost:11434)", app.model),
                Style::default().fg(GRAY),
            )),
            3 => row.push_span(Span::styled(
                format!("Workspace: {}", app.workspace.display()),
                Style::default().fg(GRAY),
            )),
            _ => {}
        }

        lines.push(row);
    }

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
        let (prefix, content_style) = match message.role {
            MessageRole::User => (
                "> ",
                Style::default().fg(DIM_GRAY).add_modifier(Modifier::BOLD),
            ),
            MessageRole::Assistant => ("", Style::default().fg(WHITE)),
        };

        for (index, line) in message.content.split('\n').enumerate() {
            let prefix_span = if index == 0 {
                Span::styled(prefix, Style::default().fg(DIM_GRAY))
            } else {
                Span::raw("  ")
            };

            lines.push(Line::from(vec![
                prefix_span,
                Span::styled(line.to_owned(), content_style),
            ]));
        }

        lines.push(Line::default());
    }

    lines
}

fn help_lines() -> Vec<Line<'static>> {
    let mut lines = vec![
        Line::from(Span::styled(
            "Echo Commands",
            Style::default().fg(WHITE).add_modifier(Modifier::BOLD),
        )),
        Line::default(),
    ];

    for (command, description) in [
        ("/help", "Show available commands and shortcuts"),
        ("/clear", "Clear the terminal screen"),
        ("/model", "Switch model: /model <name>"),
        (
            "/models",
            "List installed models and pick one (arrows + Enter)",
        ),
        ("/workspace", "Change workspace: /workspace <path>"),
        ("/stats", "Show stats from the last execution"),
        ("/tree", "Show workspace directory tree"),
        ("/ls", "List workspace directory contents"),
        ("/new", "Start a new conversation (clear history)"),
        ("/exit", "Exit Echo (alias: /quit)"),
        ("/quit", "Exit Echo (alias: /exit)"),
    ] {
        lines.push(Line::from(vec![
            Span::raw("  "),
            Span::styled(command, Style::default().fg(CYAN)),
            Span::styled(format!("   {}", description), Style::default().fg(GRAY)),
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
        ("↑ / ↓", "Navigate history"),
        ("Ctrl+R", "Reverse history search"),
        ("/", "Type to see command autocomplete"),
    ] {
        lines.push(Line::from(vec![
            Span::raw("  "),
            Span::styled(format!("{:<10}", key), Style::default().fg(CYAN)),
            Span::styled(description, Style::default().fg(GRAY)),
        ]));
    }

    lines
}

fn draw_completion(frame: &mut Frame, area: Rect, app: &EchoApp) {
    let matches = app.matching_commands();
    if matches.is_empty() {
        return;
    }

    // The completion menu is bottom-docked immediately above the input.
    // Python limits the visible menu to eight rows and shifts the visible
    // slice as the selected item moves.
    let max_rows = 8usize;
    let selected = app.completion.selected.min(matches.len() - 1);
    let start = selected.saturating_sub(max_rows - 1);
    let visible = &matches[start..matches.len().min(start + max_rows)];

    let mut rows = Vec::with_capacity(1 + visible.len());

    // One blank row separates the transcript from the completion list.
    rows.push(Line::default());

    let name_width = visible
        .iter()
        .map(|(name, _)| name.len())
        .max()
        .unwrap_or(0);

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

    frame.render_widget(Paragraph::new(rows), area);
}

fn draw_model_list(frame: &mut Frame, area: Rect, app: &EchoApp) {
    let models = ["qwen2.5:3b-instruct", "qwen2.5-coder:3b"];

    let lines = models
        .iter()
        .enumerate()
        .map(|(index, name)| {
            let selected = index == app.selected_model;
            let current = *name == app.model;
            let marker = if selected { "> " } else { "  " };
            let style = if selected {
                Style::default().fg(WHITE).add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(TEXT)
            };
            let suffix = if current { "  (current)" } else { "" };

            Line::from(Span::styled(format!("{}{}{}", marker, name, suffix), style))
        })
        .collect::<Vec<_>>();

    frame.render_widget(Paragraph::new(lines), area);
}

fn draw_composer(frame: &mut Frame, area: Rect, app: &EchoApp) {
    frame.render_widget(
        Paragraph::new(app.composer.visible_with_cursor())
            .style(Style::default().fg(WHITE))
            .wrap(Wrap { trim: false }),
        area,
    );
}

fn draw_footer(frame: &mut Frame, area: Rect, app: &EchoApp) {
    let left = "? for shortcuts";
    let right = &app.model;
    let padding = (area.width as usize)
        .saturating_sub(left.len() + right.len())
        .max(1);

    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(left, Style::default().fg(DIM_GRAY)),
            Span::raw(" ".repeat(padding)),
            Span::styled(right, Style::default().fg(DIM_GRAY)),
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
    use super::{build_middle_lines, visual_height};
    use crate::app::{EchoApp, Overlay};
    use ratatui::text::Line;

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
}
