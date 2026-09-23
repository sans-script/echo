#!/usr/bin/env python3
"""Executable entry point for Echo."""

import os
import sys

# Ensure the echo package directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from echo.cli import main

if __name__ == "__main__":
    main()
