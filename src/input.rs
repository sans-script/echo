#[derive(Debug, Default)]
pub struct Composer {
    chars: Vec<char>,
    cursor: usize,
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

    pub fn insert(&mut self, ch: char) {
        self.chars.insert(self.cursor, ch);
        self.cursor += 1;
    }

    pub fn backspace(&mut self) {
        if self.cursor > 0 {
            self.cursor -= 1;
            self.chars.remove(self.cursor);
        }
    }

    pub fn delete(&mut self) {
        if self.cursor < self.chars.len() {
            self.chars.remove(self.cursor);
        }
    }

    pub fn move_left(&mut self) {
        self.cursor = self.cursor.saturating_sub(1);
    }

    pub fn move_right(&mut self) {
        self.cursor = (self.cursor + 1).min(self.chars.len());
    }

    pub fn home(&mut self) {
        self.cursor = 0;
    }

    pub fn end(&mut self) {
        self.cursor = self.chars.len();
    }

    pub fn clear(&mut self) {
        self.chars.clear();
        self.cursor = 0;
    }

    pub fn wrapped_height(&self, width: u16) -> u16 {
        let width = usize::from(width.max(1));
        let text = self.visible_with_cursor();
        let mut rows = 0usize;

        for line in text.split('\n') {
            let line_width = line.chars().count();
            rows += line_width.div_ceil(width).max(1);
        }

        rows.max(1).min(usize::from(u16::MAX)) as u16
    }

    pub fn visible_with_cursor(&self) -> String {
        let mut output = String::with_capacity(self.chars.len() + 3);

        for (index, ch) in self.chars.iter().enumerate() {
            if index == self.cursor {
                output.push('█');
            }
            output.push(*ch);
        }

        if self.cursor == self.chars.len() {
            output.push('█');
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use super::Composer;

    #[test]
    fn inserts_and_moves_cursor() {
        let mut composer = Composer::new();
        composer.insert('a');
        composer.insert('b');
        composer.move_left();
        composer.insert('x');

        assert_eq!(composer.text(), "axb");
        assert_eq!(composer.cursor(), 2);
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
    fn cursor_rendering_has_one_cursor() {
        let mut composer = Composer::new();
        composer.insert('a');
        composer.move_left();

        assert_eq!(composer.visible_with_cursor(), "█a");
    }
}
