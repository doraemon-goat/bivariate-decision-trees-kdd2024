"""
main.py

Single entry point for the project. Running this script executes the
full experimental pipeline (Step 8): trains the baseline and
bivariate trees on the same Iris train/test split, evaluates both,
saves numeric results to results/ and plots to plots/, and prints a
summary to the terminal.

Usage:
    python main.py
"""

from src.experiment import run_experiment, print_summary

if __name__ == "__main__":
    results = run_experiment()
    print_summary(results)
