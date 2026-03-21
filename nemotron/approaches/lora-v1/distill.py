"""
Nemotron Reasoning Challenge — Training Data Generator

Generates training data in 3 tiers:
  A: Direct answers (no reasoning)
  B: Programmatic reasoning traces (unit_conversion, gravity, roman_numerals)
  C: Gemini reasoning traces (bit_manipulation, encryption, transformation)

Usage:
  python distill.py --tier A          # quick baseline
  python distill.py --tier B          # add programmatic reasoning
  python distill.py --tier C          # add Gemini reasoning for hard categories
  python distill.py --tier ABC        # all tiers (default)
"""

import argparse
import csv
import json
import os
import re
import sys
import time
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
    "Please solve this step by step. Put your final answer in \\boxed{{}}.\n"
    "<|im_end|>\n"
    "<|im_start|>assistant\n"
    "<think>\n{reasoning}\n</think>\n"
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
# Tier A — Direct answers
# ---------------------------------------------------------------------------

def tier_a(row: dict) -> dict:
    return {
        "id": row["id"],
        "prompt": row["prompt"],
        "response": CHAT_TEMPLATE.format(
            prompt=row["prompt"],
            reasoning="",
            answer=row["answer"],
        ),
        "tier": "A",
        "category": row["category"],
    }

# ---------------------------------------------------------------------------
# Tier B — Programmatic reasoning traces
# ---------------------------------------------------------------------------

def solve_unit_conversion(prompt: str, answer: str) -> str | None:
    """Extract examples, compute ratio, apply to query."""
    examples = re.findall(r"([\d.]+)\s*m\s+becomes\s+([\d.]+)", prompt)
    if not examples:
        return None

    ratios = []
    for inp, out in examples:
        inp_f, out_f = float(inp), float(out)
        if inp_f == 0:
            continue
        ratios.append(out_f / inp_f)

    if not ratios:
        return None

    avg_ratio = sum(ratios) / len(ratios)

    query_match = re.search(r"convert the following measurement:\s*([\d.]+)\s*m", prompt)
    if not query_match:
        return None
    query = float(query_match.group(1))
    computed = query * avg_ratio
    computed_str = f"{computed:.2f}"

    try:
        if abs(float(computed_str) - float(answer)) > 0.05:
            return None
        computed_str = answer  # use ground truth for final step
    except ValueError:
        return None

    lines = ["Let me find the conversion factor from the examples.\n"]
    for i, (inp, out) in enumerate(examples, 1):
        r = float(out) / float(inp)
        lines.append(f"Example {i}: {out} / {inp} = {r:.4f}")
    lines.append(f"\nAverage conversion factor: {avg_ratio:.4f}")
    lines.append(f"\nApplying to query: {query} × {avg_ratio:.4f} = {computed_str}")
    return "\n".join(lines)


def solve_gravity(prompt: str, answer: str) -> str | None:
    """Extract examples, solve for g, apply to query."""
    examples = re.findall(r"t\s*=\s*([\d.]+)\s*s.*?distance\s*=\s*([\d.]+)\s*m", prompt)
    if not examples:
        return None

    g_values = []
    for t_str, d_str in examples:
        t, d = float(t_str), float(d_str)
        if t == 0:
            continue
        g = 2 * d / (t * t)
        g_values.append(g)

    if not g_values:
        return None

    avg_g = sum(g_values) / len(g_values)

    query_match = re.search(r"falling distance for t\s*=\s*([\d.]+)\s*s", prompt)
    if not query_match:
        return None
    query_t = float(query_match.group(1))
    computed = 0.5 * avg_g * query_t * query_t
    computed_str = f"{computed:.2f}"

    try:
        if abs(float(computed_str) - float(answer)) > 0.05:
            return None
        computed_str = answer  # use ground truth for final step
    except ValueError:
        return None

    lines = ["Using the formula d = 0.5 × g × t², I'll find g from each example.\n"]
    for i, (t_str, d_str) in enumerate(examples, 1):
        t, d = float(t_str), float(d_str)
        g = 2 * d / (t * t)
        lines.append(f"Example {i}: g = 2 × {d_str} / {t_str}² = {g:.4f}")
    lines.append(f"\nAverage g = {avg_g:.4f}")
    lines.append(f"\nFor t = {query_t}s: d = 0.5 × {avg_g:.4f} × {query_t}² = {computed_str}")
    return "\n".join(lines)


def solve_roman_numerals(prompt: str, answer: str) -> str | None:
    """Roman numeral conversion — verify answer and provide reasoning."""
    query_match = re.search(r"write the number\s+(\d+)", prompt.lower())
    if not query_match:
        return None
    num = int(query_match.group(1))

    vals = [
        (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
        (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
        (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"),
    ]
    result = ""
    remaining = num
    steps = []
    for value, numeral in vals:
        while remaining >= value:
            result += numeral
            remaining -= value
            steps.append(f"{numeral} ({value})")

    if result != answer.strip():
        return None

    lines = [f"Converting {num} to Roman numerals.\n"]
    lines.append(f"Breaking down: {' + '.join(steps)}")
    lines.append(f"Result: {result}")
    return "\n".join(lines)


TIER_B_SOLVERS = {
    "unit_conversion": solve_unit_conversion,
    "gravity": solve_gravity,
    "roman_numerals": solve_roman_numerals,
}


def tier_b(row: dict) -> dict | None:
    solver = TIER_B_SOLVERS.get(row["category"])
    if solver is None:
        return None
    reasoning = solver(row["prompt"], row["answer"])
    if reasoning is None:
        return None
    return {
        "id": row["id"],
        "prompt": row["prompt"],
        "response": CHAT_TEMPLATE.format(
            prompt=row["prompt"],
            reasoning=reasoning,
            answer=row["answer"],
        ),
        "tier": "B",
        "category": row["category"],
    }

# ---------------------------------------------------------------------------
# Tier C — Gemini reasoning traces
# ---------------------------------------------------------------------------

GEMINI_PROMPT = """You are solving a puzzle from "Alice's Wonderland". Analyze the examples carefully and solve the problem step by step.

{prompt}

Think step by step. At the end, state your final answer clearly."""


def call_gemini(prompt: str, api_key: str, model: str = "gemini-2.5-pro") -> str | None:
    """Call Gemini API and return the response text."""
    try:
        import google.generativeai as genai
    except ImportError:
        print("ERROR: pip install google-generativeai", file=sys.stderr)
        sys.exit(1)

    genai.configure(api_key=api_key)
    gen_model = genai.GenerativeModel(model)
    try:
        response = gen_model.generate_content(
            GEMINI_PROMPT.format(prompt=prompt),
            generation_config=genai.GenerationConfig(
                temperature=0.2,
                max_output_tokens=2048,
            ),
        )
        return response.text
    except Exception as e:
        print(f"  Gemini error: {e}", file=sys.stderr)
        return None


def extract_answer_from_gemini(text: str) -> str | None:
    """Try to extract the final answer from Gemini's response."""
    m = re.search(r"\\boxed\{(.+?)\}", text)
    if m:
        return m.group(1).strip()
    m = re.search(r"(?:final answer|answer)\s*(?:is)?[:\s]+(.+?)\.?\s*$", text, re.IGNORECASE | re.MULTILINE)
    if m:
        return m.group(1).strip()
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    if lines:
        return lines[-1].strip(". ")
    return None


def answers_match(gemini_ans: str, ground_truth: str) -> bool:
    """Check if Gemini's answer matches the ground truth."""
    if gemini_ans is None:
        return False
    ga = gemini_ans.strip().lower()
    gt = ground_truth.strip().lower()
    if ga == gt:
        return True
    try:
        if abs(float(ga) - float(gt)) < 0.05:
            return True
    except ValueError:
        pass
    if gt in ga:
        return True
    return False


def tier_c(row: dict, api_key: str) -> dict | None:
    """Generate reasoning via Gemini, validate against ground truth."""
    response_text = call_gemini(row["prompt"], api_key)
    if response_text is None:
        return None

    gemini_answer = extract_answer_from_gemini(response_text)
    if not answers_match(gemini_answer, row["answer"]):
        print(f"  MISMATCH id={row['id']}: gemini='{gemini_answer}' vs truth='{row['answer']}'")
        return None

    return {
        "id": row["id"],
        "prompt": row["prompt"],
        "response": CHAT_TEMPLATE.format(
            prompt=row["prompt"],
            reasoning=response_text,
            answer=row["answer"],
        ),
        "tier": "C",
        "category": row["category"],
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate training data for Nemotron")
    parser.add_argument("--tier", default="AB", help="Which tiers to generate: A, B, C, or combinations like AB, ABC")
    parser.add_argument("--gemini-model", default="gemini-2.5-pro", help="Gemini model to use for Tier C")
    parser.add_argument("--max-c", type=int, default=0, help="Max Tier C samples (0=all)")
    args = parser.parse_args()

    tiers = set(args.tier.upper())

    print(f"Loading training data from {TRAIN_CSV}...")
    rows = load_train_data()
    print(f"Loaded {len(rows)} samples")

    from collections import Counter
    cat_counts = Counter(r["category"] for r in rows)
    for cat, count in sorted(cat_counts.items()):
        print(f"  {cat}: {count}")

    # Gemini API key for Tier C
    api_key = None
    if "C" in tiers:
        env_path = _ROOT / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if "GEMINI_API_KEY" in line:
                    api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
        if not api_key:
            api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("ERROR: No Gemini API key found in .env or environment", file=sys.stderr)
            sys.exit(1)

    tier_b_cats = set(TIER_B_SOLVERS.keys())
    tier_c_cats = {"bit_manipulation", "encryption", "transformation"}

    results = {}
    stats = {"A": 0, "B": 0, "B_fail": 0, "C": 0, "C_fail": 0, "C_skip": 0}

    # --- Tier B ---
    if "B" in tiers:
        print("\n=== Tier B: Programmatic reasoning ===")
        for row in rows:
            if row["category"] in tier_b_cats:
                result = tier_b(row)
                if result:
                    results[row["id"]] = result
                    stats["B"] += 1
                else:
                    stats["B_fail"] += 1
        print(f"  Generated: {stats['B']}, Failed: {stats['B_fail']}")

    # --- Tier C ---
    if "C" in tiers:
        print("\n=== Tier C: Gemini reasoning ===")
        c_candidates = [r for r in rows if r["category"] in tier_c_cats and r["id"] not in results]
        if args.max_c > 0:
            c_candidates = c_candidates[:args.max_c]
        print(f"  Candidates: {len(c_candidates)}")
        for i, row in enumerate(c_candidates):
            print(f"  [{i+1}/{len(c_candidates)}] id={row['id']} cat={row['category']}...", end=" ", flush=True)
            result = tier_c(row, api_key)
            if result:
                results[row["id"]] = result
                stats["C"] += 1
                print("OK")
            else:
                stats["C_fail"] += 1
                print("FAIL")
            if i < len(c_candidates) - 1:
                time.sleep(4)

    # --- Tier A — fill in anything not covered by B or C ---
    if "A" in tiers:
        print("\n=== Tier A: Direct answers (filling gaps) ===")
        for row in rows:
            if row["id"] not in results:
                results[row["id"]] = tier_a(row)
                stats["A"] += 1
        print(f"  Generated: {stats['A']}")

    # --- Write output ---
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Writing {len(results)} samples to {OUTPUT_FILE} ===")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for rid in sorted(results.keys()):
            f.write(json.dumps(results[rid], ensure_ascii=False) + "\n")

    print("\nFinal stats:")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")
    print(f"  Total: {len(results)}")


if __name__ == "__main__":
    main()
