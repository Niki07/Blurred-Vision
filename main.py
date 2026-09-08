"""Entry point for a real experiment run.

Usage:
    python main.py motion
    python main.py defocus
    python main.py glass
    python main.py          # defaults to motion
"""

import sys

from src.pipeline import run_blur_experiment

if __name__ == "__main__":
    run_blur_experiment(sys.argv[1] if len(sys.argv) > 1 else "motion")
