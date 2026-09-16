"""
Root wrapper for ASL-SR-DPT benchmark runner.
Allows running `python benchmark_runner.py` directly from project root.
"""

import sys
import os

# Add code/ to sys.path
sys.path.insert(0, os.path.abspath("code"))

from benchmark_runner import parse_args, run_benchmark

if __name__ == "__main__":
    args = parse_args()
    run_benchmark(args)
