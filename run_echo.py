import sys
sys.stdout.write("[2J[H")
sys.stdout.flush()

#!/usr/bin/env python3
"""Executable entry point for Echo."""

import os
import sys

# Ensure the echo package directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sys
sys.stdout.write("[3J[2J[H")
sys.stdout.flush()

from echo.cli import main

if __name__ == "__main__":
    main()
