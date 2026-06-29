#!/usr/bin/env python
"""Thin wrapper: `python scripts/run_sim.py` == `rtb sim`."""
from rtb_rl.cli import sim_cmd

if __name__ == "__main__":
    sim_cmd()
