#!/usr/bin/env python
"""Thin wrapper: `python scripts/serve.py` == `rtb serve`."""
from rtb_rl.cli import serve_cmd

if __name__ == "__main__":
    serve_cmd()
