//! "⠋ Loading..." indicator shown while Echo waits for the model, with the
//! moving highlight from the Python version.

use std::time::{Duration, Instant};

pub const SPINNER_FRAMES: [char; 10] = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
pub const FRAME_TIME: Duration = Duration::from_millis(140);
pub const LOADING_TEXT: &str = "Loading...";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shade {
    /// The highlighted character.
    Bright,
    White,
    Light,
    Gray,
}

pub fn frame_index(since: Instant) -> usize {
    (since.elapsed().as_millis() / FRAME_TIME.as_millis()) as usize
}

/// Spinner glyph plus each character of the text with its shade.
pub fn loading_frame(frame: usize) -> (char, Vec<(char, Shade)>) {
    let chars = LOADING_TEXT.chars().collect::<Vec<_>>();
    let position = frame % chars.len();
    let shaded = chars
        .iter()
        .enumerate()
        .map(|(index, ch)| {
            let shade = match index.abs_diff(position) {
                0 => Shade::Bright,
                1 => Shade::White,
                2 => Shade::Light,
                _ => Shade::Gray,
            };
            (*ch, shade)
        })
        .collect();
    (SPINNER_FRAMES[frame % SPINNER_FRAMES.len()], shaded)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn highlight_moves_with_the_frame() {
        let (glyph, shaded) = loading_frame(3);
        assert_eq!(glyph, '⠸');
        assert_eq!(shaded[3].1, Shade::Bright);
        assert_eq!(shaded[2].1, Shade::White);
        assert_eq!(shaded[0].1, Shade::Gray);
    }
}
