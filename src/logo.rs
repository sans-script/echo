//! Live logo animation for the banner, ported from `LogoAnimator` in the
//! Python version. Instead of a background thread, the animator is advanced
//! from the UI loop with `tick`, which keeps rendering on a single thread.

use crate::logo_frames::{ECHO_LOGO, ECHO_LOGO_FRAMES, LOGO_HEIGHT};
use std::{
    collections::VecDeque,
    f64::consts::TAU,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

pub const LOGO_FPS: u32 = 30;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogoTone {
    White,
    Gray,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LogoFrame {
    pub lines: Vec<String>,
    pub tone: LogoTone,
}

impl LogoFrame {
    fn of(lines: &[&str]) -> Self {
        Self {
            lines: lines.iter().map(|line| line.to_string()).collect(),
            tone: LogoTone::White,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Effect {
    Rotation,
    Particles,
}

pub struct LogoAnimator {
    enabled: bool,
    rng: Rng,
    queue: VecDeque<(LogoFrame, Duration)>,
    current: LogoFrame,
    next_at: Instant,
    last: Option<Effect>,
}

impl LogoAnimator {
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            rng: Rng::from_clock(),
            queue: VecDeque::new(),
            current: LogoFrame::of(&ECHO_LOGO),
            next_at: Instant::now(),
            last: None,
        }
    }

    pub fn current(&self) -> &LogoFrame {
        &self.current
    }

    /// Advances to the frame due at `now`. Returns whether it changed.
    pub fn tick(&mut self, now: Instant) -> bool {
        if !self.enabled || now < self.next_at {
            return false;
        }
        if self.queue.is_empty() {
            self.schedule_next_effect();
        }
        let Some((frame, duration)) = self.queue.pop_front() else {
            return false;
        };
        self.next_at = now + duration;
        let changed = frame != self.current;
        self.current = frame;
        changed
    }

    fn schedule_next_effect(&mut self) {
        let effect = match self.last {
            Some(Effect::Rotation) => Effect::Particles,
            Some(Effect::Particles) => Effect::Rotation,
            None if self.rng.next_f64() < 0.5 => Effect::Rotation,
            None => Effect::Particles,
        };
        self.last = Some(effect);

        let frame_time = Duration::from_secs_f64(1.0 / f64::from(LOGO_FPS));
        let frames = match effect {
            Effect::Rotation => ECHO_LOGO_FRAMES
                .iter()
                .map(|frame| LogoFrame::of(frame))
                .collect(),
            Effect::Particles => self.particles(),
        };
        self.queue
            .extend(frames.into_iter().map(|frame| (frame, frame_time)));

        // Short random rest on the static logo between effects.
        let rest = Duration::from_secs_f64(self.rng.uniform(0.08, 0.18));
        self.queue.push_back((LogoFrame::of(&ECHO_LOGO), rest));
    }

    /// Particles disperse from the logo, orbit around its center and then
    /// converge back into it.
    fn particles(&mut self) -> Vec<LogoFrame> {
        let points = logo_points();
        let count = points.len() as f64;
        let cx = points.iter().map(|(x, _)| x).sum::<f64>() / count;
        let cy = points.iter().map(|(_, y)| y).sum::<f64>() / count;

        struct Particle {
            x: f64,
            y: f64,
            angle: f64,
            radius: f64,
            speed: f64,
            phase: f64,
            drift: f64,
        }
        let data = points
            .iter()
            .map(|&(x, y)| Particle {
                x,
                y,
                angle: (y - cy).atan2(x - cx),
                radius: self.rng.uniform(3.0, 7.0),
                speed: self.rng.uniform(0.7, 1.3),
                phase: self.rng.uniform(0.0, TAU),
                drift: self.rng.uniform(0.10, 0.40),
            })
            .collect::<Vec<_>>();

        let smooth = |t: f64| t * t * (3.0 - 2.0 * t);
        let mut frames = vec![LogoFrame::of(&ECHO_LOGO); 8];

        for i in 0..26 {
            let t = smooth(f64::from(i) / 25.0);
            let step = f64::from(i);
            let positions = data.iter().map(|p| {
                let radius = p.radius * t;
                let tx = cx + p.angle.cos() * radius + (step * 0.17 + p.phase).sin() * p.drift * t;
                let ty = cy
                    + p.angle.sin() * radius * 0.65
                    + (step * 0.13 + p.phase).cos() * p.drift * t;
                (p.x * (1.0 - t) + tx * t, p.y * (1.0 - t) + ty * t)
            });
            frames.push(particle_frame(positions, 1.0 - t * 0.25));
        }

        for i in 0..42 {
            let step = f64::from(i);
            let positions = data.iter().map(|p| {
                let angle = p.angle + step * 0.025 * p.speed;
                let radius = p.radius + (step * 0.10 + p.phase).sin() * 0.45;
                (
                    cx + angle.cos() * radius + (step * 0.07 + p.phase).sin() * 0.20,
                    cy + angle.sin() * radius * 0.65 + (step * 0.11 + p.phase).cos() * 0.20,
                )
            });
            frames.push(particle_frame(positions, 0.85));
        }

        for i in 0..30 {
            let t = smooth(f64::from(i) / 29.0);
            let positions = data.iter().map(|p| {
                let angle = p.angle + 42.0 * 0.025 * p.speed;
                let radius = p.radius * (1.0 - t);
                let ox = cx + angle.cos() * radius;
                let oy = cy + angle.sin() * radius * 0.65;
                (ox * (1.0 - t) + p.x * t, oy * (1.0 - t) + p.y * t)
            });
            frames.push(particle_frame(positions, 0.55 + t * 0.45));
        }

        frames.extend(vec![LogoFrame::of(&ECHO_LOGO); 8]);
        frames
    }
}

fn logo_points() -> Vec<(f64, f64)> {
    ECHO_LOGO
        .iter()
        .enumerate()
        .flat_map(|(y, row)| {
            row.chars()
                .enumerate()
                .filter(|(_, ch)| *ch == ':')
                .map(move |(x, _)| (x as f64, y as f64))
        })
        .collect()
}

fn particle_frame(positions: impl Iterator<Item = (f64, f64)>, brightness: f64) -> LogoFrame {
    let width = ECHO_LOGO[0].len();
    let mut grid = vec![vec![' '; width]; LOGO_HEIGHT];
    for (x, y) in positions {
        let (ix, iy) = (x.round(), y.round());
        if ix >= 0.0 && iy >= 0.0 && (ix as usize) < width && (iy as usize) < LOGO_HEIGHT {
            grid[iy as usize][ix as usize] = ':';
        }
    }
    LogoFrame {
        lines: grid.into_iter().map(|row| row.into_iter().collect()).collect(),
        tone: if brightness >= 0.80 {
            LogoTone::White
        } else {
            LogoTone::Gray
        },
    }
}

/// Small xorshift generator; the animation only needs visual randomness.
struct Rng(u64);

impl Rng {
    fn from_clock() -> Self {
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0x9E37_79B9_7F4A_7C15);
        Self(seed | 1)
    }

    fn next_f64(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }

    fn uniform(&mut self, low: f64, high: f64) -> f64 {
        low + (high - low) * self.next_f64()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_animator_keeps_static_logo() {
        let mut animator = LogoAnimator::new(false);
        assert!(!animator.tick(Instant::now() + Duration::from_secs(5)));
        assert_eq!(animator.current().lines, ECHO_LOGO);
    }

    #[test]
    fn effects_alternate_and_return_to_the_logo() {
        let mut animator = LogoAnimator::new(true);
        let mut now = Instant::now();
        let mut effects = Vec::new();
        for _ in 0..4 {
            animator.tick(now);
            effects.push(animator.last);
            // Drain the effect, ending on the resting logo frame.
            while !animator.queue.is_empty() {
                now += Duration::from_secs(1);
                animator.tick(now);
            }
            assert_eq!(animator.current().lines, ECHO_LOGO);
            now += Duration::from_secs(1);
        }
        assert_ne!(effects[0], effects[1]);
        assert_ne!(effects[1], effects[2]);
    }

    #[test]
    fn particle_frames_keep_logo_dimensions() {
        let mut animator = LogoAnimator::new(true);
        for frame in animator.particles() {
            assert_eq!(frame.lines.len(), LOGO_HEIGHT);
            assert!(frame.lines.iter().all(|line| line.len() == ECHO_LOGO[0].len()));
        }
    }
}
