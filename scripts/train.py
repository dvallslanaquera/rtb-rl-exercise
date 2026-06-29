#!/usr/bin/env python
"""Thin wrapper: `python scripts/train.py` == `rtb train`."""
from rtb_rl.cli import train_cmd

if __name__ == "__main__":
    train_cmd()
