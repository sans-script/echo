from pathlib import Path

p = Path("echo/cli.py")
text = p.read_text(encoding="utf-8")

lines = text.splitlines()

lines[93] = 'SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"'

p.write_text("\n".join(lines) + "\n", encoding="utf-8")

print("SPINNER_FRAMES corrigido.")