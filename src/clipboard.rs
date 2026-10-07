//! Copies text selected in the TUI to the system clipboard without extra
//! dependencies: `clip.exe` on Windows (works in any console), and the
//! OSC 52 terminal escape sequence elsewhere.

use std::io::Write;

pub fn copy(terminal: &mut impl Write, text: &str) {
    #[cfg(windows)]
    if copy_with_clip_exe(text).is_ok() {
        return;
    }
    let _ = write!(terminal, "\x1b]52;c;{}\x07", base64(text.as_bytes()));
    let _ = terminal.flush();
}

/// `clip.exe` reads UTF-16 when the input starts with a byte order mark,
/// which keeps accented characters intact.
#[cfg(windows)]
fn copy_with_clip_exe(text: &str) -> std::io::Result<()> {
    use std::process::{Command, Stdio};

    let mut child = Command::new("clip.exe")
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()?;
    let mut bytes = vec![0xFF, 0xFE];
    bytes.extend(text.encode_utf16().flat_map(u16::to_le_bytes));
    child
        .stdin
        .take()
        .ok_or_else(|| std::io::Error::other("clip.exe has no stdin"))?
        .write_all(&bytes)?;
    let status = child.wait()?;
    if status.success() {
        Ok(())
    } else {
        Err(std::io::Error::other("clip.exe failed"))
    }
}

fn base64(bytes: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let n = chunk
            .iter()
            .enumerate()
            .fold(0u32, |n, (i, byte)| n | (u32::from(*byte) << (16 - 8 * i)));
        for i in 0..4 {
            if i <= chunk.len() {
                out.push(TABLE[(n >> (18 - 6 * i) & 63) as usize] as char);
            } else {
                out.push('=');
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::base64;

    #[test]
    fn base64_matches_the_standard_encoding() {
        assert_eq!(base64(b""), "");
        assert_eq!(base64(b"f"), "Zg==");
        assert_eq!(base64(b"fo"), "Zm8=");
        assert_eq!(base64(b"foo"), "Zm9v");
        assert_eq!(base64("ação".as_bytes()), "YcOnw6Nv");
    }
}
