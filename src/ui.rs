use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Borders, Clear, Paragraph, Wrap},
    Frame,
};

use crate::{
    app::{EchoApp, MessageRole, Overlay},
};

const DIVIDER: &str = "─";

pub fn draw(frame: &mut Frame, app: &EchoApp) {
    let area = frame.area();

    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(5),
            Constraint::Min(1),
            Constraint::Length(2),
            Constraint::Length(1),
        ])
        .split(area);

    draw_header(frame, layout[0], app);
    draw_chat(frame, layout[1], app);
    draw_composer(frame, layout[2], app);
    draw_footer(frame, layout[3], app);

    match app.overlay {
        Overlay::None => {}
        Overlay::Help => draw_help(frame, area),
        Overlay::Commands => draw_commands(frame, area, app),
    }
}

fn draw_header(frame: &mut Frame, area: Rect, app: &EchoApp) {
    let lines = vec![
        Line::from("Echo CLI 0.1.0"),
        Line::from("Local AI Support Assistant"),
        Line::from(format!("Model: {}", app.model)),
        Line::from(format!("Workspace: {}", app.workspace.display())),
        Line::from(DIVIDER.repeat(area.width as usize)),
    ];

    frame.render_widget(
        Paragraph::new(Text::from(lines)).style(Style::default().fg(Color::White)),
        area,
    );
}

fn draw_chat(frame: &mut Frame, area: Rect, app: &EchoApp) {
    let mut lines = Vec::new();

    for message in &app.messages {
        let prefix = match message.role {
            MessageRole::User => "> ",
            MessageRole::Assistant => "  ",
        };

        for (index, line) in message.content.lines().enumerate() {
            if index == 0 {
                lines.push(Line::from(vec![
                    Span::styled(prefix, Style::default().add_modifier(Modifier::BOLD)),
                    Span::raw(line),
                ]));
            } else {
                lines.push(Line::from(format!("  {line}")));
            }
        }

        lines.push(Line::from(""));
    }

    if lines.is_empty() {
        lines.push(Line::from(""));
    }

    let content_height = area.height as usize;
    let max_scroll = lines.len().saturating_sub(content_height);
    let scroll = app.scroll.min(max_scroll) as u16;

    frame.render_widget(
        Paragraph::new(Text::from(lines))
            .scroll((scroll, 0))
            .wrap(Wrap { trim: false }),
        area,
    );
}

fn draw_composer(frame: &mut Frame, area: Rect, app: &EchoApp) {
    let text = format!("> {}", app.composer.visible_with_cursor());

    frame.render_widget(
        Paragraph::new(text)
            .style(Style::default().fg(Color::White))
            .block(Block::default()),
        area,
    );
}

fn draw_footer(frame: &mut Frame, area: Rect, app: &EchoApp) {
    let footer = format!("? for shortcuts{}", " ".repeat(
        area.width
            .saturating_sub(1)
            .saturating_sub(format!("? for shortcuts").len()),
    ));

    let model = format!("{} ", app.model);
    let mut spans = vec![Span::raw(footer)];

    if model.len() < area.width as usize {
        let start = area.width as usize - model.len();
        let text = format!("{}{}", " ".repeat(start.saturating_sub("? for shortcuts".len())), model);
        spans = vec![Span::raw("? for shortcuts"), Span::raw(text)];
    }

    frame.render_widget(Paragraph::new(Line::from(spans)), area);
}

fn draw_help(frame: &mut Frame, area: Rect) {
    let popup = centered_rect(70, 60, area);

    frame.render_widget(Clear, popup);
    frame.render_widget(
        Paragraph::new(vec![
            Line::from("Echo shortcuts"),
            Line::from(""),
            Line::from("Enter       Submit prompt"),
            Line::from("Left/Right  Move cursor"),
            Line::from("Home/End    Move to line boundary"),
            Line::from("Up/Down     Scroll chat"),
            Line::from("?           Toggle shortcuts"),
            Line::from("/           Open commands"),
            Line::from("Esc         Close overlay"),
            Line::from("Ctrl+C      Exit"),
        ])
        .block(Block::bordered().title(" Help ")),
        popup,
    );
}

fn draw_commands(frame: &mut Frame, area: Rect, _app: &EchoApp) {
    let popup = centered_rect(60, 45, area);

    frame.render_widget(Clear, popup);
    frame.render_widget(
        Paragraph::new(vec![
            Line::from("/help      Show shortcuts"),
            Line::from("/models    Open model picker"),
            Line::from("/new       Start a new conversation"),
            Line::from("/clear     Clear chat"),
            Line::from("/quit      Exit Echo"),
        ])
        .block(Block::bordered().title(" Commands ")),
        popup,
    );
}

fn centered_rect(percent_x: u16, percent_y: u16, area: Rect) -> Rect {
    let vertical = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(area);

    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(vertical[1])[1]
}

#[cfg(test)]
mod tests {
    use super::centered_rect;
    use ratatui::layout::Rect;

    #[test]
    fn popup_stays_inside_terminal() {
        let area = Rect::new(0, 0, 120, 40);
        let popup = centered_rect(60, 50, area);

        assert!(popup.x >= area.x);
        assert!(popup.y >= area.y);
        assert!(popup.right() <= area.right());
        assert!(popup.bottom() <= area.bottom());
    }
}
