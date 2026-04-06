"""
run_pipeline.py
===============
Convenience script that runs all pipeline steps in order.

Usage
-----
    # Full pipeline (may take several hours for scraping)
    python run_pipeline.py

    # Skip scraping (if data already collected)
    python run_pipeline.py --skip-scrape

    # Run only specific steps
    python run_pipeline.py --steps 2 3 4 5 6

    # Show step descriptions
    python run_pipeline.py --list
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# Ensure child processes print UTF-8 on Windows
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

STEPS = {
    1: ("01_scrape_talks.py",      "Scrape all General Conference talks (1971–2025)"),
    2: ("02_preprocess.py",        "Clean text, build master DataFrame"),
    3: ("03_tithing_detect.py",    "Keyword + embedding tithing detection"),
    4: ("04_topic_model.py",       "NMF topic modelling (full + tithing corpus)"),
    5: ("05_temporal_analysis.py", "Temporal aggregation & statistical tests"),
    6: ("06_visualize.py",         "Generate all figures"),
}

SCRIPT_DIR = Path(__file__).parent


def run_step(step_num: int) -> bool:
    script, desc = STEPS[step_num]
    print(f"\n{'='*65}")
    print(f"STEP {step_num}: {desc}")
    print(f"{'='*65}")
    t0 = time.time()

    result = subprocess.run(
        [sys.executable, str(SCRIPT_DIR / script)],
        cwd=SCRIPT_DIR,
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n[ERROR] Step {step_num} failed (exit code {result.returncode}).")
        return False

    print(f"\n[OK] Step {step_num} completed in {elapsed:.0f}s")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run the Tithing Discourse Analysis pipeline."
    )
    parser.add_argument(
        "--skip-scrape",
        action="store_true",
        help="Skip step 1 (scraping). Use when talk data already exists.",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        type=int,
        choices=list(STEPS.keys()),
        help="Run only specific step numbers (e.g. --steps 3 4 5).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all pipeline steps and exit.",
    )
    args = parser.parse_args()

    if args.list:
        print("Pipeline steps:")
        for num, (script, desc) in STEPS.items():
            print(f"  {num}. {desc}  [{script}]")
        return

    steps_to_run = args.steps if args.steps else list(STEPS.keys())
    if args.skip_scrape and 1 in steps_to_run:
        steps_to_run.remove(1)

    print(f"Running steps: {steps_to_run}")
    total_t0 = time.time()

    for step in sorted(steps_to_run):
        ok = run_step(step)
        if not ok:
            print(f"\nPipeline aborted at step {step}.")
            sys.exit(1)

    elapsed = time.time() - total_t0
    print(f"\n{'='*65}")
    print(f"All steps completed in {elapsed/60:.1f} min.")
    print(f"Results: output/figures/  and  output/tables/")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
