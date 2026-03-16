"""
inference.py — LoRA Qwen2.5-7B Inference
Loads fine-tuned LoRA adapters + score head, runs ensemble over folds.
Designed to run on Kaggle with no internet access (P100 GPU).

Expected dataset layout (/kaggle/input/llm-qlora-finetuned/):
    qlora_lora_f0/adapter_config.json + adapter_model.safetensors
    qlora_lora_f1/...
    best_qlora_score_f0.pt
    best_qlora_score_f1.pt
    tokenizer_config.json + tokenizer.json + ...

NOTE: Verify BASE_MODEL_DIR mount path before first Kaggle run.
      Check kaggle.com/models/qwen-lm/qwen2.5 → Files tab → exact path.
"""

import contextlib
import glob
import json
import os
import re
import unicodedata
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
transformers.logging.set_verbosity_error()
from peft import PeftModel
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
KAGGLE = os.path.exists("/kaggle")

BASE_MODEL_DIR = (
    "/kaggle/input/models/qwen-lm/qwen2.5/transformers/7b/1"
    if KAGGLE else
    "Qwen/Qwen2.5-7B"                           # uses HF cache locally
)

_HERE       = os.path.dirname(os.path.abspath(__file__))
ADAPTER_DIR = (
    "/kaggle/input/llm-qlora-finetuned"
    if KAGGLE else
    os.path.normpath(os.path.join(_HERE, "..", "..", "kaggle_dataset", "qlora-finetuned"))
)
DATA_DIR = (
    "/kaggle/input/competitions/llm-classification-finetuning"
    if KAGGLE else
    os.path.normpath(os.path.join(_HERE, "..", "..", "llm-classification-finetuning"))
)
OUTPUT = "/kaggle/working/submission.csv" if KAGGLE else "submission.csv"

MAX_LEN       = 2048
BATCH_SIZE    = 2
# P100 (Pascal) has no bfloat16 hardware support; use float16 on Kaggle
COMPUTE_DTYPE = torch.float16 if KAGGLE else torch.bfloat16

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(
    f"CUDA available: {torch.cuda.is_available()}"
    + (f"  |  GPU: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "  |  Running on CPU")
)
print(f"Device: {DEVICE}  |  dtype: {COMPUTE_DTYPE}  |  Adapter dir: {ADAPTER_DIR}")

LABEL_COLS = ["winner_model_a", "winner_model_b", "winner_tie"]


# ---------------------------------------------------------------------------
# Helpers (verbatim from train.py)
# ---------------------------------------------------------------------------

def parse_prompt(s: str) -> str:
    try:
        parts = json.loads(s)
        if isinstance(parts, list):
            return "\n".join(str(p) for p in parts)
    except Exception:
        pass
    return str(s)


def clean_text(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()


def truncate_parts(tokenizer, prompt: str, resp_a: str, resp_b: str, max_len: int):
    budget = max_len - 4
    p_ids = tokenizer.encode(prompt, add_special_tokens=False)
    a_ids = tokenizer.encode(resp_a,  add_special_tokens=False)
    b_ids = tokenizer.encode(resp_b,  add_special_tokens=False)
    total = len(p_ids) + len(a_ids) + len(b_ids)
    if total > budget:
        ratio = budget / total
        p_ids = p_ids[:max(1, int(len(p_ids) * ratio))]
        a_ids = a_ids[:max(1, int(len(a_ids) * ratio))]
        b_ids = b_ids[:max(1, int(len(b_ids) * ratio))]
    p = tokenizer.decode(p_ids, skip_special_tokens=True)
    a = tokenizer.decode(a_ids, skip_special_tokens=True)
    b = tokenizer.decode(b_ids, skip_special_tokens=True)
    return p, a, b


def build_prompt(prompt: str, resp_a: str, resp_b: str) -> str:
    return (
        "Evaluate two AI responses and classify which is better.\n\n"
        f"Prompt:\n{prompt}\n\n"
        f"Response A:\n{resp_a}\n\n"
        f"Response B:\n{resp_b}\n\n"
        "Which response is better? A, B, or Tie."
    )


# ---------------------------------------------------------------------------
# Dataset & Collator
# ---------------------------------------------------------------------------

class QLoraDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer):
        self.records = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Tokenizing", leave=False):
            prompt = clean_text(parse_prompt(row["prompt"]))
            resp_a = clean_text(parse_prompt(row["response_a"]))
            resp_b = clean_text(parse_prompt(row["response_b"]))
            prompt_t, resp_a_t, resp_b_t = truncate_parts(
                tokenizer, prompt, resp_a, resp_b, MAX_LEN
            )
            text = build_prompt(prompt_t, resp_a_t, resp_b_t)
            enc  = tokenizer(text, max_length=MAX_LEN, truncation=True, padding=False)
            self.records.append({
                "input_ids":      torch.tensor(enc["input_ids"],      dtype=torch.long),
                "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            })

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return dict(self.records[idx])


def make_collate_fn(pad_token_id: int):
    def collate(batch: list) -> dict:
        max_len = max(item["input_ids"].size(0) for item in batch)
        input_ids_list, attn_mask_list = [], []
        for item in batch:
            pad_len = max_len - item["input_ids"].size(0)
            input_ids_list.append(torch.cat([
                torch.full((pad_len,), pad_token_id, dtype=torch.long),
                item["input_ids"],
            ]))
            attn_mask_list.append(torch.cat([
                torch.zeros(pad_len, dtype=torch.long),
                item["attention_mask"],
            ]))
        return {
            "input_ids":      torch.stack(input_ids_list),
            "attention_mask": torch.stack(attn_mask_list),
        }
    return collate


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def load_inference_model(adapter_subdir: str, score_path: str):
    """Load base model in float16/bfloat16, attach LoRA adapters, restore score head."""
    base_model = AutoModelForSequenceClassification.from_pretrained(
        Path(BASE_MODEL_DIR),
        num_labels=3,
        torch_dtype=COMPUTE_DTYPE,
        device_map={"": 0},
        attn_implementation="sdpa",
    )
    if base_model.config.pad_token_id is None:
        base_model.config.pad_token_id = base_model.config.eos_token_id

    # adapter_config.json records training-time base path; mismatch is expected and safe
    model = PeftModel.from_pretrained(base_model, Path(adapter_subdir))

    # Restore score head (PEFT's save_pretrained omits non-LoRA modules)
    score_sd = torch.load(score_path, map_location=DEVICE, weights_only=True)
    model.base_model.model.score.load_state_dict(score_sd)

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict(model, loader) -> np.ndarray:
    all_logits = []
    for batch in tqdm(loader, desc="Inference", leave=False):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        ctx = (
            torch.autocast(device_type="cuda", dtype=COMPUTE_DTYPE)
            if DEVICE.type == "cuda"
            else contextlib.nullcontext()
        )
        with ctx:
            outputs = model(**batch)
        all_logits.append(outputs.logits.cpu().float())
    return torch.cat(all_logits).softmax(-1).numpy()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    print(f"Test rows: {len(test_df)}")

    adapter_dirs = sorted(glob.glob(os.path.join(ADAPTER_DIR, "qlora_lora_f*")))
    score_paths  = sorted(glob.glob(os.path.join(ADAPTER_DIR, "best_qlora_score_f*.pt")))

    if not adapter_dirs:
        raise FileNotFoundError(f"No LoRA adapter dirs found in {ADAPTER_DIR}")
    if len(adapter_dirs) != len(score_paths):
        raise ValueError(
            f"Adapter/score count mismatch: {len(adapter_dirs)} adapters vs {len(score_paths)} score heads"
        )
    print(f"Found {len(adapter_dirs)} fold checkpoint(s)")

    # Tokenizer saved into ADAPTER_DIR by save.py
    tokenizer = AutoTokenizer.from_pretrained(Path(ADAPTER_DIR), use_fast=True)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "left"
    collate_fn = make_collate_fn(tokenizer.pad_token_id)

    test_ds     = QLoraDataset(test_df, tokenizer)
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, collate_fn=collate_fn,
    )

    all_preds = []
    for adapter_dir, score_path in zip(adapter_dirs, score_paths):
        fold_name = os.path.basename(adapter_dir)
        print(f"\n=== {fold_name} ===")
        model = load_inference_model(adapter_dir, score_path)
        preds = predict(model, test_loader)
        all_preds.append(preds)
        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    final_preds = np.mean(all_preds, axis=0)

    sub = pd.DataFrame(final_preds, columns=LABEL_COLS)
    sub.insert(0, "id", test_df["id"].values)
    sub.to_csv(OUTPUT, index=False)
    print(f"\nSubmission saved to {OUTPUT}")
    print(sub.head())
    assert (sub[LABEL_COLS].sum(axis=1) - 1.0).abs().max() < 1e-4, "Probabilities don't sum to 1!"
    print("Probability sum check passed.")


if __name__ == "__main__":
    main()
