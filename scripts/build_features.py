#!/usr/bin/env python
"""Thin wrapper: `python scripts/build_features.py` == `rtb build-features`."""
from rtb_rl.cli import build_features_cmd

if __name__ == "__main__":
    build_features_cmd()
