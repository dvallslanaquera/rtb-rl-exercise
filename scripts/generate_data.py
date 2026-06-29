#!/usr/bin/env python
"""Thin wrapper: `python scripts/generate_data.py` == `rtb generate-data`."""
from rtb_rl.cli import generate_data

if __name__ == "__main__":
    generate_data()
