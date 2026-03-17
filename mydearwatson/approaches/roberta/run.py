#!/usr/bin/env python3
"""
Deploy RoBERTa approach: train -> save -> upload dataset -> push kernel -> poll status.

Usage:
    python run.py [--skip-train] [--message "my message"]
"""

import argparse
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

DATASET_DIR = Path(__file__).parent.parent.parent / "kaggle_dataset" / "watson-roberta-finetuned"
KERNEL_ID = "lasseruttert/watson-roberta-inference"
SCRIPT_DIR = Path(__file__).parent
COMPETITION = "contradictory-my-dear-watson"


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
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument(
        "--message",
        default=f"auto deploy roberta {datetime.now().strftime('%Y-%m-%d %H:%M')}",
    )
    args = parser.parse_args()

    if not args.skip_train:
        header("Training")
        conda_check(["--no-capture-output", "python", "-u", str(SCRIPT_DIR / "train.py")])

    header("Saving model")
    conda_check(["--no-capture-output", "python", "-u", str(SCRIPT_DIR / "save.py")])

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

    header("Pushing kernel")
    conda_check(["kaggle", "kernels", "push", "-p", str(SCRIPT_DIR)])

    header("Polling kernel status")
    status = ""
    while not re.search(r"complete|error|cancel", status, re.IGNORECASE):
        time.sleep(30)
        result = conda(["kaggle", "kernels", "status", KERNEL_ID], capture_output=True, text=True)
        status = result.stdout.strip()
        print(status)

    download_cmd = (
        f"cd '{SCRIPT_DIR}' && conda run -n kaggle "
        f"kaggle kernels output {KERNEL_ID} -p ./output --force"
    )

    if re.search(r"complete", status, re.IGNORECASE):
        print("\n=== Done! ===")
        print(f"  Check results at: https://www.kaggle.com/competitions/{COMPETITION}/submissions")
        print(f"\n  To download kernel output locally:\n    {download_cmd}")
    else:
        print(f"\n=== Kernel ended with status: {status} ===")
        print(f"  To fetch output for debugging:\n    {download_cmd}")


if __name__ == "__main__":
    main()
