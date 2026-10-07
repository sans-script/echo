use std::time::{Duration, Instant};

const CURSOR_BLINK_INTERVAL: Duration = Duration::from_millis(530);
const CURSOR_TYPING_HOLD: Duration = Duration::from_millis(550);

#[derive(Debug, Clone)]
struct CursorBlink {
    visible: bool,
    typing_until: Option<Instant>,
    next_toggle: Instant,
}

impl Default for CursorBlink {
    fn default() -> Self {
        let now = Instant::now();
        Self {
            visible: true,
            typing_until: None,
            next_toggle: now + CURSOR_BLINK_INTERVAL,
        }
    }
}

impl CursorBlink {
    fn mark_typing(&mut self) {
        let now = Instant::now();
        self.visible = true;
        self.typing_until = Some(now + CURSOR_TYPING_HOLD);
        self.next_toggle = now + CURSOR_TYPING_HOLD + CURSOR_BLINK_INTERVAL;
    }

    fn force_visible(&mut self) {
        let now = Instant::now();
        self.visible = true;
        self.typing_until = None;
        self.next_toggle = now + CURSOR_BLINK_INTERVAL;
    }

    fn tick(&mut self, now: Instant) {
        if let Some(typing_until) = self.typing_until {
            if now < typing_until {
                // Restart the blink schedule while editing, exactly like the
                // Python FakeCursorBlink implementation.
                self.visible = true;
                self.next_toggle = now + CURSOR_BLINK_INTERVAL;
                return;
            }

            self.typing_until = None;
        }

        if now >= self.next_toggle {
            self.visible = !self.visible;
            self.next_toggle = now + CURSOR_BLINK_INTERVAL;
        }
    }
}

#[derive(Debug, Default)]
pub struct Composer {
    chars: Vec<char>,
    cursor: usize,
    blink: CursorBlink,
}

impl Composer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn text(&self) -> String {
        self.chars.iter().collect()
    }

    pub fn is_empty(&self) -> bool {
        self.chars.is_empty()
    }

    pub fn cursor(&self) -> usize {
        self.cursor
    }

    pub fn cursor_visible(&self) -> bool {
        self.blink.visible
    }

    pub fn tick_cursor(&mut self) {
        self.blink.tick(Instant::now());
    }

    pub fn force_cursor_visible(&mut self) {
        self.blink.force_visible();
    }

    fn mark_cursor_activity(&mut self) {
        self.blink.mark_typing();
    }

    pub fn insert(&mut self, ch: char) {
        self.chars.insert(self.cursor, ch);
        self.cursor += 1;
        self.mark_cursor_activity();
    }

    pub fn backspace(&mut self) {
        if self.cursor > 0 {
            self.cursor -= 1;
            self.chars.remove(self.cursor);
            self.mark_cursor_activity();
        }
    }

    pub fn delete(&mut self) {
        if self.cursor < self.chars.len() {
            self.chars.remove(self.cursor);
            self.mark_cursor_activity();
        }
    }

    pub fn move_left(&mut self) {
        let old = self.cursor;
        self.cursor = self.cursor.saturating_sub(1);
        if self.cursor != old {
            self.mark_cursor_activity();
        }
    }

    pub fn move_right(&mut self) {
        let old = self.cursor;
        self.cursor = (self.cursor + 1).min(self.chars.len());
        if self.cursor != old {
            self.mark_cursor_activity();
        }
    }

    pub fn home(&mut self) {
        let old = self.cursor;
        self.cursor = 0;
        if self.cursor != old {
            self.mark_cursor_activity();
        }
    }

    pub fn end(&mut self) {
        let old = self.cursor;
        self.cursor = self.chars.len();
        if self.cursor != old {
            self.mark_cursor_activity();
        }
    }

    pub fn clear(&mut self) {
        self.chars.clear();
        self.cursor = 0;
        self.force_cursor_visible();
    }

    pub fn insert_str(&mut self, text: &str) {
        for ch in text.chars() {
            self.chars.insert(self.cursor, ch);
            self.cursor += 1;
        }
        self.mark_cursor_activity();
    }

    pub fn wrapped_height(&self, width: u16) -> u16 {
        let width = usize::from(width.max(1));
        // The software cursor can take one extra column at the end of the text.
        let text = self.text();
        let lines = text.split('\n').collect::<Vec<_>>();
        let rows = lines
            .iter()
            .enumerate()
            .map(|(index, line)| {
                let cursor = usize::from(index + 1 == lines.len());
                (line.chars().count() + cursor).div_ceil(width).max(1)
            })
            .sum::<usize>();

        rows.max(1).min(usize::from(u16::MAX)) as u16
    }

    pub fn visible_with_cursor(&self) -> String {
        self.render_with_cursor(self.blink.visible)
    }

    pub fn render_with_cursor(&self, cursor_visible: bool) -> String {
        let mut output = String::with_capacity(self.chars.len() + 1);

        for (index, ch) in self.chars.iter().enumerate() {
            if cursor_visible && index == self.cursor {
                output.push('█');
            }
            output.push(*ch);
        }

        if cursor_visible && self.cursor == self.chars.len() {
            output.push('█');
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use super::{CURSOR_BLINK_INTERVAL, CURSOR_TYPING_HOLD, Composer, CursorBlink};
    use std::time::{Duration, Instant};

    #[test]
    fn inserts_and_moves_cursor() {
        let mut composer = Composer::new();
        composer.insert('a');
        composer.insert('b');
        composer.move_left();
        composer.insert('x');

        assert_eq!(composer.text(), "axb");
        assert_eq!(composer.cursor(), 2);
        assert!(composer.cursor_visible());
    }

    #[test]
    fn backspace_and_delete_are_cursor_aware() {
        let mut composer = Composer::new();
        composer.insert('a');
        composer.insert('b');
        composer.insert('c');

        composer.move_left();
        composer.backspace();
        assert_eq!(composer.text(), "ac");

        composer.delete();
        assert_eq!(composer.text(), "a");
    }

    #[test]
    fn height_counts_explicit_line_breaks() {
        let mut composer = Composer::new();
        composer.insert_str("one\ntwo\nthree");

        assert_eq!(composer.wrapped_height(80), 3);
        assert_eq!(composer.wrapped_height(3), 4);
    }

    #[test]
    fn cursor_rendering_has_one_cursor() {
        let mut composer = Composer::new();
        composer.insert('a');
        composer.move_left();

        assert_eq!(composer.visible_with_cursor(), "█a");
        assert_eq!(composer.render_with_cursor(false), "a");
    }

    #[test]
    fn typing_hold_keeps_cursor_solid() {
        let mut blink = CursorBlink::default();
        let start = Instant::now();

        blink.typing_until = Some(start + CURSOR_TYPING_HOLD);
        blink.next_toggle = start;
        blink.tick(start + Duration::from_millis(100));

        assert!(blink.visible);
        assert!(blink.next_toggle > start + CURSOR_TYPING_HOLD);
    }

    #[test]
    fn cursor_blinks_after_typing_hold() {
        let mut blink = CursorBlink::default();
        let start = Instant::now();

        blink.visible = true;
        blink.typing_until = Some(start + CURSOR_TYPING_HOLD);
        blink.next_toggle = start + CURSOR_TYPING_HOLD + CURSOR_BLINK_INTERVAL;

        blink.tick(start + CURSOR_TYPING_HOLD + CURSOR_BLINK_INTERVAL);
        assert!(!blink.visible);

        blink.tick(start + CURSOR_TYPING_HOLD + CURSOR_BLINK_INTERVAL * 2);
        assert!(blink.visible);
    }
}
