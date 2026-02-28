#!/usr/bin/env python3
"""
Runner for cuda-path-to-flash exercises.

Usage:
    python run.py 01/01          # run module 01, exercise 01
    python run.py 04/03          # run module 04, exercise 03
    python run.py 01             # run all exercises in module 01
    python run.py all            # run everything
    python run.py 01/01 --bench  # run with benchmarks enabled

Works on Kaggle, Colab, or any machine with PyTorch + CUDA.
No nvcc needed — PyTorch compiles kernels on the fly.
"""

import argparse
import importlib
import sys
import os
from pathlib import Path

def discover_exercises(base: Path, module_filter=None, exercise_filter=None):
    """Find all exercise .py files matching the filter."""
    exercises = []
    for mod_dir in sorted(base.iterdir()):
        if not mod_dir.is_dir() or not mod_dir.name[:2].isdigit():
            continue
        mod_num = mod_dir.name[:2]
        if module_filter and mod_num != module_filter:
            continue

        ex_dir = mod_dir / "exercises"
        if not ex_dir.exists():
            continue

        for ex_file in sorted(ex_dir.glob("*.py")):
            ex_num = ex_file.stem.split("_")[0]
            if exercise_filter and ex_num != exercise_filter:
                continue
            exercises.append(ex_file)

    return exercises

def run_exercise(path: Path, bench=False):
    """Import and run a single exercise."""
    # Add parent to path so imports work
    sys.path.insert(0, str(path.parent))

    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)

    # Set bench flag so exercises can check it
    os.environ["CUDA_BENCH"] = "1" if bench else "0"

    print(f"\n{'='*60}")
    print(f"  {path.parent.parent.name}/{path.parent.name}/{path.name}")
    print(f"{'='*60}\n")

    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

    sys.path.pop(0)

def main():
    parser = argparse.ArgumentParser(description="Run CUDA exercises")
    parser.add_argument("target", nargs="?", default="all",
                       help="Module/exercise to run: '01/02', '01', or 'all'")
    parser.add_argument("--bench", action="store_true",
                       help="Enable benchmarks")
    args = parser.parse_args()

    base = Path(__file__).parent

    if args.target == "all":
        mod_filter, ex_filter = None, None
    elif "/" in args.target:
        mod_filter, ex_filter = args.target.split("/")
    else:
        mod_filter, ex_filter = args.target, None

    exercises = discover_exercises(base, mod_filter, ex_filter)

    if not exercises:
        print(f"No exercises found for '{args.target}'")
        sys.exit(1)

    print(f"Found {len(exercises)} exercise(s)")

    for ex in exercises:
        run_exercise(ex, args.bench)

if __name__ == "__main__":
    main()
