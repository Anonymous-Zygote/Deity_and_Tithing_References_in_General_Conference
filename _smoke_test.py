"""
Quick smoke-test: scrapes 2023–2024 general conferences then runs
preprocess → detect → topic model → temporal → visualize.

Temporarily patches config.py's year range, then restores it.
"""
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
PYTHON = sys.executable
CONFIG = ROOT / "config.py"
BACKUP = ROOT / "_config_backup.py"

STEPS = [
    ("01_scrape_talks.py",      "Scrape talks"),
    ("02_preprocess.py",        "Preprocess"),
    ("03_tithing_detect.py",    "Detect tithing"),
    ("04_topic_model.py",       "Topic model"),
    ("05_temporal_analysis.py", "Temporal analysis"),
    ("06_visualize.py",         "Visualize"),
]


def patch_config(start: int, end: int):
    """Overwrite START_YEAR / END_YEAR in config.py."""
    text = CONFIG.read_text(encoding="utf-8")
    import re
    text = re.sub(r"^START_YEAR\s*=\s*\d+", f"START_YEAR = {start}", text, flags=re.M)
    text = re.sub(r"^END_YEAR\s*=\s*\d+",   f"END_YEAR   = {end}",   text, flags=re.M)
    CONFIG.write_text(text, encoding="utf-8")


def run(script: str, label: str) -> bool:
    print(f"\n{'='*55}")
    print(f"  {label}  [{script}]")
    print(f"{'='*55}")
    t0 = time.time()
    r = subprocess.run([PYTHON, str(ROOT / script)], cwd=ROOT)
    elapsed = time.time() - t0
    ok = r.returncode == 0
    status = "OK" if ok else "FAILED"
    print(f"  [{status}] in {elapsed:.1f}s")
    return ok


def main():
    # Backup original config
    shutil.copy(CONFIG, BACKUP)
    try:
        # Limit to 2 conferences for a fast test
        patch_config(2023, 2024)

        for script, label in STEPS:
            if not run(script, label):
                print("\n✗ Smoke test FAILED.")
                return

        print("\n✓ All steps passed – smoke test complete.")

    finally:
        # Always restore original config
        shutil.copy(BACKUP, CONFIG)
        BACKUP.unlink(missing_ok=True)
        print("  config.py restored.")


if __name__ == "__main__":
    main()

