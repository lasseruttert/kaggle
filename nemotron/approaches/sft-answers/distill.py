"""
Nemotron Reasoning Challenge — Training Data Generator (Answers Only)

Generates training data with direct answers only — no reasoning traces.
All 9500 samples get the same simple template: prompt → \boxed{answer}.

Usage:
  python distill.py
"""

import csv
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent
_ROOT = _HERE.parent.parent
DATA_DIR = _ROOT / "nvidia-nemotron-model-reasoning-challenge"
TRAIN_CSV = DATA_DIR / "train.csv"
OUTPUT_FILE = _ROOT / "kaggle_dataset" / "nemotron-training-data" / "training_data.jsonl"

CHAT_TEMPLATE = (
    "<|im_start|>user\n{prompt}\n"
    "Put your final answer in \\boxed{{}}.\n"
    "<|im_end|>\n"
    "<|im_start|>assistant\n"
    "The answer is \\boxed{{{answer}}}<|im_end|>"
)

# ---------------------------------------------------------------------------
# Category detection
# ---------------------------------------------------------------------------

def classify(prompt: str) -> str:
    pl = prompt.lower()
    if "bit manipulation" in pl:
        return "bit_manipulation"
    if "encryption" in pl or "cipher" in pl or "decrypt" in pl:
        return "encryption"
    if "gravity" in pl or "gravitational" in pl:
        return "gravity"
    if "unit conversion" in pl or "conversion" in pl and "numeral" not in pl:
        return "unit_conversion"
    if "numeral" in pl:
        return "roman_numerals"
    if "transformation" in pl:
        return "transformation"
    return "unknown"

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_train_data() -> list[dict]:
    rows = []
    with open(TRAIN_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["category"] = classify(row["prompt"])
            rows.append(row)
    return rows

# ---------------------------------------------------------------------------
# Format sample (direct answer only)
# ---------------------------------------------------------------------------

def format_sample(row: dict) -> dict:
    return {
        "id": row["id"],
        "prompt": row["prompt"],
        "response": CHAT_TEMPLATE.format(
            prompt=row["prompt"],
            answer=row["answer"],
        ),
        "tier": "A",
        "category": row["category"],
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    from collections import Counter

    print(f"Loading training data from {TRAIN_CSV}...")
    rows = load_train_data()
    print(f"Loaded {len(rows)} samples")

    cat_counts = Counter(r["category"] for r in rows)
    for cat, count in sorted(cat_counts.items()):
        print(f"  {cat}: {count}")

    results = [format_sample(row) for row in rows]

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting {len(results)} samples to {OUTPUT_FILE}")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("Done.")


if __name__ == "__main__":
    main()
