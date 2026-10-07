//! Renders the model's markdown with terminal colors in the transcript:
//! headings, lists, quotes, highlighted code blocks and inline `code`, **bold**,
//! *italic* and [links](url). It works line by line, so a reply that is
//! still streaming renders correctly as it grows; markers that are not
//! closed yet are shown as typed.

use crate::highlight::Highlighter;
use ratatui::{
    style::{Color, Modifier, Style},
    text::Span,
};

const TEXT: Color = Color::Rgb(255, 255, 255);
const GRAY: Color = Color::Rgb(128, 128, 128);
const ACCENT: Color = Color::Cyan;
const CODE: Color = Color::Rgb(229, 192, 123);
const LINK: Color = Color::Rgb(97, 175, 239);
const GUTTER: Color = Color::Rgb(95, 95, 95);

/// Rendered lines of `text`. Code block fences (```java ... ```) are not
/// shown; the code between them is syntax highlighted instead.
pub fn render(text: &str) -> Vec<Vec<Span<'static>>> {
    let base = Style::default().fg(TEXT);
    let mut code_block: Option<(Highlighter, Vec<Vec<Span<'static>>>)> = None;
    let mut lines = Vec::new();

    for line in text.split('\n') {
        if let Some(info) = line.trim_start().strip_prefix("```") {
            match code_block.take() {
                Some((_, code)) => lines.extend(numbered(code)),
                None => code_block = Some((Highlighter::new(info), Vec::new())),
            }
            continue;
        }
        match code_block.as_mut() {
            Some((highlighter, code)) => code.push(highlighter.line(line)),
            None => lines.push(block_line(line, base)),
        }
    }
    // A block still streaming in (no closing fence yet).
    if let Some((_, code)) = code_block {
        lines.extend(numbered(code));
    }

    // Keep one (empty) line so the reply prefix still has a row to sit on
    // while only an opening fence has streamed in.
    if lines.is_empty() {
        lines.push(Vec::new());
    }
    lines
}

/// Prefixes each code line with its number in a muted gutter ("12 │ ").
fn numbered(code: Vec<Vec<Span<'static>>>) -> impl Iterator<Item = Vec<Span<'static>>> {
    let width = code.len().to_string().len();
    let gutter = Style::default().fg(GUTTER);
    code.into_iter().enumerate().map(move |(index, spans)| {
        let mut line = vec![Span::styled(format!("{:>width$} │ ", index + 1), gutter)];
        line.extend(spans);
        line
    })
}

fn block_line(line: &str, base: Style) -> Vec<Span<'static>> {
    let indent_len = line.len() - line.trim_start().len();
    let (indent, rest) = line.split_at(indent_len);

    // Headings: "# Title" ... "###### Title".
    let hashes = rest.chars().take_while(|ch| *ch == '#').count();
    if (1..=6).contains(&hashes) && rest[hashes..].starts_with(' ') {
        let heading = Style::default().fg(ACCENT).add_modifier(Modifier::BOLD);
        return inline(rest[hashes + 1..].trim(), heading);
    }

    // Horizontal rule.
    let compact = rest.replace(' ', "");
    if compact.len() >= 3 && (compact.chars().all(|c| c == '-') || compact.chars().all(|c| c == '*')) {
        return vec![Span::styled("─".repeat(24), Style::default().fg(GRAY))];
    }

    // Quote.
    if let Some(quote) = rest.strip_prefix("> ").or(rest.strip_prefix(">").filter(|q| q.is_empty())) {
        let mut spans = vec![Span::styled(format!("{indent}│ "), Style::default().fg(GRAY))];
        spans.extend(inline(quote, Style::default().fg(GRAY).add_modifier(Modifier::ITALIC)));
        return spans;
    }

    // Bullets "- ", "* ", "+ " and numbered items "1. ".
    let marker = ["- ", "* ", "+ "]
        .iter()
        .find(|marker| rest.starts_with(**marker))
        .map(|marker| (marker.len(), "• ".to_string()))
        .or_else(|| {
            let digits = rest.chars().take_while(char::is_ascii_digit).count();
            (digits > 0 && rest[digits..].starts_with(". "))
                .then(|| (digits + 2, rest[..digits + 2].to_string()))
        });
    if let Some((len, shown)) = marker {
        let mut spans = vec![
            Span::raw(indent.to_string()),
            Span::styled(shown, Style::default().fg(ACCENT)),
        ];
        spans.extend(inline(&rest[len..], base));
        return spans;
    }

    let mut spans = vec![Span::raw(indent.to_string())];
    spans.extend(inline(rest, base));
    spans
}

/// Inline markup inside one line. A marker only applies when its closing
/// marker is on the same line.
fn inline(text: &str, base: Style) -> Vec<Span<'static>> {
    let mut spans = Vec::new();
    let mut plain = String::new();
    let mut rest = text;

    while let Some(ch) = rest.chars().next() {
        let styled = match ch {
            '`' => closed(rest, "`").map(|(inner, used)| {
                (vec![Span::styled(inner.to_string(), Style::default().fg(CODE))], used)
            }),
            '*' if rest.starts_with("**") => closed(rest, "**").map(|(inner, used)| {
                (inline(inner, base.add_modifier(Modifier::BOLD)), used)
            }),
            '*' => closed(rest, "*")
                .filter(|(inner, _)| !inner.starts_with(' ') && !inner.ends_with(' '))
                .map(|(inner, used)| (inline(inner, base.add_modifier(Modifier::ITALIC)), used)),
            '[' => link(rest).map(|(label, used)| {
                let style = Style::default().fg(LINK).add_modifier(Modifier::UNDERLINED);
                (vec![Span::styled(label.to_string(), style)], used)
            }),
            _ => None,
        };

        match styled {
            Some((inner, used)) => {
                if !plain.is_empty() {
                    spans.push(Span::styled(std::mem::take(&mut plain), base));
                }
                spans.extend(inner);
                rest = &rest[used..];
            }
            None => {
                plain.push(ch);
                rest = &rest[ch.len_utf8()..];
            }
        }
    }
    if !plain.is_empty() || spans.is_empty() {
        spans.push(Span::styled(plain, base));
    }
    spans
}

/// `marker inner marker` at the start of `text`: the inner text and the
/// number of bytes used, when the marker is closed and the inner is not empty.
fn closed<'a>(text: &'a str, marker: &str) -> Option<(&'a str, usize)> {
    let body = &text[marker.len()..];
    let end = body.find(marker)?;
    (end > 0).then(|| (&body[..end], marker.len() * 2 + end))
}

/// `[label](url)` at the start of `text`.
fn link(text: &str) -> Option<(&str, usize)> {
    let close = text.find("](")?;
    let label = &text[1..close];
    let url_end = text[close + 2..].find(')')?;
    (!label.is_empty() && !label.contains('[')).then(|| (label, close + 3 + url_end))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text_of(spans: &[Span]) -> String {
        spans.iter().map(|span| span.content.as_ref()).collect()
    }

    fn style_of<'a>(spans: &'a [Span], content: &str) -> &'a Style {
        &spans.iter().find(|span| span.content == content).unwrap().style
    }

    #[test]
    fn inline_markers_are_styled_and_hidden() {
        let line = &render("Use **cargo build** and `ls -la` *now*")[0];
        assert_eq!(text_of(line), "Use cargo build and ls -la now");
        assert!(style_of(line, "cargo build").add_modifier.contains(Modifier::BOLD));
        assert_eq!(style_of(line, "ls -la").fg, Some(CODE));
        assert!(style_of(line, "now").add_modifier.contains(Modifier::ITALIC));
    }

    #[test]
    fn unclosed_markers_stay_as_typed_while_streaming() {
        assert_eq!(text_of(&render("a **partial")[0]), "a **partial");
        assert_eq!(text_of(&render("2 * 3 * 4")[0]), "2 * 3 * 4");
        assert_eq!(text_of(&render("snake_case_name")[0]), "snake_case_name");
    }

    #[test]
    fn lists_headings_and_links() {
        let lines = render("## Files\n- **a.txt**\n2. second\nSee [docs](https://x.y)");
        assert_eq!(text_of(&lines[0]), "Files");
        assert_eq!(lines[0][0].style.fg, Some(ACCENT));
        assert_eq!(text_of(&lines[1]), "• a.txt");
        assert_eq!(text_of(&lines[2]), "2. second");
        assert_eq!(text_of(&lines[3]), "See docs");
    }

    #[test]
    fn code_blocks_hide_fences_and_are_highlighted() {
        let lines = render("Code:\n```rust\nlet x = **y**;\n```\nafter **b**");
        assert_eq!(lines.len(), 3);
        assert_eq!(text_of(&lines[1]), "1 │ let x = **y**;");
        assert_eq!(lines[1][0].style.fg, Some(GUTTER));
        assert_eq!(
            lines[1][1].style,
            crate::highlight::style(crate::highlight::Token::Keyword)
        );
        assert_eq!(text_of(&lines[2]), "after b");
    }

    #[test]
    fn code_line_numbers_are_aligned_per_block() {
        let code = (1..=10).map(|n| format!("x{n}")).collect::<Vec<_>>().join("\n");
        let lines = render(&format!("```\n{code}\n```\n```\nsecond\n```"));
        assert_eq!(text_of(&lines[0]), " 1 │ x1");
        assert_eq!(text_of(&lines[9]), "10 │ x10");
        // Each block starts again at 1.
        assert_eq!(text_of(&lines[10]), "1 │ second");
    }

    #[test]
    fn a_streaming_open_fence_still_renders_a_line() {
        assert_eq!(render("```java").len(), 1);
        let lines = render("```java\npublic class Main {");
        assert_eq!(text_of(&lines[0]), "1 │ public class Main {");
    }
}
