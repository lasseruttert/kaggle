#!/usr/bin/env python3
"""
Deploy sft-answers: distill data → upload dataset → push kernel → poll status.

Usage:
    python run.py                  # full pipeline
    python run.py --skip-distill   # reuse existing training_data.jsonl
"""

import argparse
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

_HERE = Path(__file__).parent
_ROOT = _HERE.parent.parent
DATASET_DIR = _ROOT / "kaggle_dataset" / "nemotron-training-data"
KERNEL_ID = "lasseruttert/nemotron-sft-answers"
COMPETITION = "nvidia-nemotron-3-reasoning-challenge"


def conda(args: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(["conda", "run", "-n", "kaggle"] + args, **kwargs)


def conda_check(args: list[str], **kwargs) -> subprocess.CompletedProcess:
    result = conda(args, **kwargs)
    if result.returncode != 0:
        sys.exit(result.returncode)
    return result


def header(msg: str) -> None:
    print(f"\n=== {msg} ===")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-distill", action="store_true", help="Skip data generation, reuse existing data")
    parser.add_argument("--message", default=f"auto deploy sft-answers {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    args = parser.parse_args()

    # --- Step 1: Generate training data ---
    if not args.skip_distill:
        header("Distilling training data (answers only)")
        conda_check(["--no-capture-output", "python", "-u", str(_HERE / "distill.py")])

    # Verify data exists
    data_file = DATASET_DIR / "training_data.jsonl"
    if not data_file.exists():
        print(f"ERROR: {data_file} not found. Run without --skip-distill first.", file=sys.stderr)
        sys.exit(1)
    size_mb = data_file.stat().st_size / 1024 / 1024
    print(f"Training data: {data_file} ({size_mb:.1f} MB)")

    # --- Step 2: Upload dataset ---
    header("Uploading dataset")
    result = conda(
        ["kaggle", "datasets", "version", "-p", str(DATASET_DIR), "-m", args.message],
        capture_output=True, text=True,
    )
    new_dataset = False
    if result.returncode != 0:
        print("Dataset not found, creating...")
        conda_check(["kaggle", "datasets", "create", "-p", str(DATASET_DIR)])
        new_dataset = True

    wait_secs = 90 if new_dataset else 30
    header(f"Waiting {wait_secs}s for dataset to be processed by Kaggle")
    time.sleep(wait_secs)

    # --- Step 3: Push kernel ---
    header("Pushing kernel")
    conda_check(["kaggle", "kernels", "push", "-p", str(_HERE), "--accelerator", "NvidiaRtxPro6000"])

    # --- Step 4: Poll kernel status ---
    header("Polling kernel status")
    status = ""
    while not re.search(r"complete|error|cancel", status, re.IGNORECASE):
        time.sleep(30)
        result = conda(["kaggle", "kernels", "status", KERNEL_ID], capture_output=True, text=True)
        status = result.stdout.strip()
        print(status)

    download_cmd = f"cd '{_HERE}' && conda run -n kaggle kaggle kernels output {KERNEL_ID} -p ./output --force"

    if re.search(r"complete", status, re.IGNORECASE):
        print("\n=== Done! ===")
        print("  This is a code competition - submission is auto-scored on kernel complete.")
        print(f"  Check results at: https://www.kaggle.com/competitions/{COMPETITION}/submissions")
        print(f"\n  To download kernel output locally:\n    {download_cmd}")
    else:
        print(f"\n=== Kernel ended with status: {status} ===")
        print(f"  To fetch output for debugging:\n    {download_cmd}")


if __name__ == "__main__":
    main()
