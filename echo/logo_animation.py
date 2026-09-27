"""Echo logo animations — rotation, particles, materialize, glitch.

Each function renders straight to the terminal using cursor-addressing
escape codes and blocks for its own duration. They're meant to be called
directly (e.g. from a startup flourish or the /logo slash command), not
run in an infinite loop.
"""

import sys
import time
import random
import math

from .logo_frames import ECHO_LOGO, ECHO_LOGO_FRAMES

FPS_ROTATION = 30
FPS_EFFECTS = 30

WHITE = "\033[38;2;255;255;255m"
GRAY = "\033[38;2;150;150;150m"
DARK = "\033[38;2;70;70;70m"
RESET = "\033[0m"

WIDTH = len(ECHO_LOGO[0])
HEIGHT = len(ECHO_LOGO)

POINTS = [
    (x, y)
    for y, row in enumerate(ECHO_LOGO)
    for x, char in enumerate(row)
    if char == ":"
]


def _clear():
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()


def _render(frame):
    sys.stdout.write("\033[H")
    sys.stdout.write("\n".join(frame))
    sys.stdout.write("\033[J")
    sys.stdout.flush()


def _wait(fps):
    time.sleep(1.0 / fps)


def _normal_frame():
    return [
        "".join(f"{WHITE}:{RESET}" if c == ":" else c for c in row)
        for row in ECHO_LOGO
    ]


def rotation(loops: int = 1):
    """Spin the logo through ECHO_LOGO_FRAMES `loops` times."""
    for _ in range(loops):
        for frame in ECHO_LOGO_FRAMES:
            colored = [
                "".join(f"{WHITE}:{RESET}" if c == ":" else c for c in row)
                for row in frame
            ]
            _render(colored)
            _wait(FPS_ROTATION)


def _render_particle_frame(positions, brightness):
    grid = [[" " for _ in range(WIDTH)] for _ in range(HEIGHT)]
    for x, y in positions:
        x, y = round(x), round(y)
        if 0 <= x < WIDTH and 0 <= y < HEIGHT:
            grid[y][x] = ":"

    color = WHITE if brightness >= 0.8 else GRAY if brightness >= 0.55 else DARK
    _render([f"{color}{''.join(row)}{RESET}" for row in grid])


def particles():
    cx = sum(x for x, y in POINTS) / len(POINTS)
    cy = sum(y for x, y in POINTS) / len(POINTS)
    rng = random.Random()

    data = [
        {
            "x": x, "y": y,
            "angle": math.atan2(y - cy, x - cx),
            "radius": rng.uniform(3.0, 7.0),
            "speed": rng.uniform(0.7, 1.3),
            "phase": rng.uniform(0, math.tau),
            "drift": rng.uniform(0.1, 0.4),
        }
        for x, y in POINTS
    ]

    for _ in range(12):
        _render(_normal_frame())
        _wait(FPS_EFFECTS)

    # dispersal
    for i in range(30):
        t = i / 29.0
        t = t * t * (3.0 - 2.0 * t)
        positions = []
        for p in data:
            radius = p["radius"] * t
            tx = cx + math.cos(p["angle"]) * radius + math.sin(i * 0.17 + p["phase"]) * p["drift"] * t
            ty = cy + math.sin(p["angle"]) * radius * 0.65 + math.cos(i * 0.13 + p["phase"]) * p["drift"] * t
            x = p["x"] * (1 - t) + tx * t
            y = p["y"] * (1 - t) + ty * t
            positions.append((x, y))
        _render_particle_frame(positions, 1.0 - t * 0.25)
        _wait(FPS_EFFECTS)

    # floating
    for i in range(55):
        positions = []
        for p in data:
            angle = p["angle"] + i * 0.025 * p["speed"]
            radius = p["radius"] + math.sin(i * 0.10 + p["phase"]) * 0.45
            x = cx + math.cos(angle) * radius + math.sin(i * 0.07 + p["phase"]) * 0.20
            y = cy + math.sin(angle) * radius * 0.65 + math.cos(i * 0.11 + p["phase"]) * 0.20
            positions.append((x, y))
        _render_particle_frame(positions, 0.85)
        _wait(FPS_EFFECTS)

    # convergence
    for i in range(34):
        t = i / 33.0
        t = t * t * (3.0 - 2.0 * t)
        positions = []
        for p in data:
            angle = p["angle"] + 55 * 0.025 * p["speed"]
            radius = p["radius"] * (1 - t)
            x = cx + math.cos(angle) * radius
            y = cy + math.sin(angle) * radius * 0.65
            x = x * (1 - t) + p["x"] * t
            y = y * (1 - t) + p["y"] * t
            positions.append((x, y))
        _render_particle_frame(positions, 0.55 + t * 0.45)
        _wait(FPS_EFFECTS)

    for _ in range(12):
        _render(_normal_frame())
        _wait(FPS_EFFECTS)


def materialize():
    rng = random.Random()
    points = POINTS[:]
    rng.shuffle(points)
    current = set()
    empty = [" " * WIDTH for _ in range(HEIGHT)]

    for _ in range(12):
        _render(empty)
        _wait(FPS_EFFECTS)

    for point in points:
        current.add(point)
        frame = []
        for y, row in enumerate(ECHO_LOGO):
            line = ""
            for x, char in enumerate(row):
                line += f"{WHITE}:{RESET}" if (char == ":" and (x, y) in current) else " "
            frame.append(line)
        _render(frame)
        _wait(FPS_EFFECTS)

    for _ in range(20):
        _render(_normal_frame())
        _wait(FPS_EFFECTS)


def _glitch_frame():
    result = []
    shift = random.choice([-3, -2, -1, 1, 2, 3])

    for row in ECHO_LOGO:
        if random.random() < 0.45:
            result.append("".join(f"{WHITE}:{RESET}" if c == ":" else c for c in row))
            continue

        shifted = [" "] * WIDTH
        for x, char in enumerate(row):
            new_x = x + shift
            if 0 <= new_x < WIDTH and char == ":":
                shifted[new_x] = ":"

        for _ in range(random.randint(1, 3)):
            x = random.randrange(WIDTH)
            if shifted[x] == ":":
                shifted[x] = random.choice(["\u00b7", "\u2591", "\u2592", "\u2593"])

        if random.random() < 0.25:
            start = random.randrange(WIDTH)
            length = random.randint(1, 4)
            for x in range(start, min(start + length, WIDTH)):
                shifted[x] = " "

        line = []
        for char in shifted:
            if char == ":":
                line.append(f"{WHITE}:{RESET}")
            elif char == "\u00b7":
                line.append(f"{GRAY}\u00b7{RESET}")
            elif char in ("\u2591", "\u2592", "\u2593"):
                line.append(f"{DARK}{char}{RESET}")
            else:
                line.append(char)
        result.append("".join(line))

    return result


def glitch(events: int = 8):
    for _ in range(25):
        _render(_normal_frame())
        _wait(FPS_EFFECTS)

    for _ in range(events):
        for _ in range(random.randint(8, 18)):
            _render(_normal_frame())
            _wait(FPS_EFFECTS)
        for _ in range(random.randint(2, 5)):
            _render(_glitch_frame())
            _wait(FPS_EFFECTS)

    for _ in range(20):
        _render(_normal_frame())
        _wait(FPS_EFFECTS)


EFFECTS = {
    "rotation": rotation,
    "particles": particles,
    "materialize": materialize,
    "glitch": glitch,
}


def play(name: str = "rotation"):
    """Run a single named effect. Hides/restores the cursor around it and
    clears the screen first so partial terminal content doesn't bleed in."""
    effect = EFFECTS.get(name, rotation)
    sys.stdout.write("\033[?25l")
    sys.stdout.flush()
    try:
        _clear()
        effect()
    finally:
        sys.stdout.write("\033[?25h")
        sys.stdout.flush()


def play_random():
    play(random.choice(list(EFFECTS)))
