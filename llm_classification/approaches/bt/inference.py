"""
inference.py — Davidson Bradley-Terry RoBERTa inference.
Loads seed checkpoints and produces submission.csv.
Designed to run on Kaggle with no internet access.

Expected Kaggle dataset layout:
  /kaggle/input/llm-bert-bt-finetuned/
      best_bt_s*.pt
      config.json
      tokenizer_config.json
      ...
"""

import glob
import json
import os
import re
import unicodedata
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
KAGGLE       = os.path.exists("/kaggle")
MODEL_DIR    = "/kaggle/input/llm-bert-bt-finetuned" if KAGGLE else "G:/My Drive/kaggle/llm_classification/kaggle_dataset/bt-finetuned"
DATA_DIR     = "/kaggle/input/competitions/llm-classification-finetuning" if KAGGLE else "G:/My Drive/kaggle/llm_classification/llm-classification-finetuning"
OUTPUT       = "/kaggle/working/submission.csv" if KAGGLE else "submission.csv"
MAX_LEN      = 512
BATCH_SIZE   = 16
PROMPT_BUDGET = 128
CKPT_PATTERN = os.path.join(MODEL_DIR, "best_bt_s*.pt")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"CUDA available: {torch.cuda.is_available()}" + (f"  |  GPU: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "  |  Running on CPU"))
print(f"Device: {DEVICE}  |  Model dir: {MODEL_DIR}")

LABEL_COLS = ["winner_model_a", "winner_model_b", "winner_tie"]


# ---------------------------------------------------------------------------
# Helpers
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


def encode_pair(tokenizer, prompt: str, response: str, max_len: int) -> dict:
    budget = max_len - 3
    prompt_budget   = min(PROMPT_BUDGET, budget // 3)
    response_budget = budget - prompt_budget
    p_ids = tokenizer.encode(prompt,   add_special_tokens=False)[:prompt_budget]
    r_ids = tokenizer.encode(response, add_special_tokens=False)[:response_budget]
    input_ids = [tokenizer.cls_token_id] + p_ids + [tokenizer.sep_token_id] + r_ids + [tokenizer.sep_token_id]
    attention_mask = [1] * len(input_ids)
    return {
        "input_ids":      torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


# ---------------------------------------------------------------------------
# Model (must match train.py exactly)
# ---------------------------------------------------------------------------

class BTRewardModel(nn.Module):
    DROPOUT = 0.1

    def __init__(self, model_dir: str):
        super().__init__()
        config = RobertaConfig.from_json_file(os.path.join(model_dir, "config.json"))
        self.backbone   = RobertaModel(config)
        self.dropout    = nn.Dropout(self.DROPOUT)
        self.score_head = nn.Linear(config.hidden_size, 1, bias=False)
        self.tie_head   = nn.Sequential(
            nn.Linear(config.hidden_size, 128),
            nn.GELU(),
            nn.Dropout(self.DROPOUT),
            nn.Linear(128, 1),
        )

    def _encode(self, input_ids, attention_mask):
        cls = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        return self.dropout(cls)

    def forward(self, a_input_ids, a_attention_mask, b_input_ids, b_attention_mask):
        cls_a = self._encode(a_input_ids, a_attention_mask)
        cls_b = self._encode(b_input_ids, b_attention_mask)
        score_a = self.score_head(cls_a).squeeze(-1)
        score_b = self.score_head(cls_b).squeeze(-1)
        tie_logit = self.tie_head(torch.abs(cls_a - cls_b)).squeeze(-1)
        logits = torch.stack([
            score_a - score_b,
            score_b - score_a,
            tie_logit,
        ], dim=1)
        return logits


# ---------------------------------------------------------------------------
# Dataset / Collator
# ---------------------------------------------------------------------------

class BTCollator:
    def __init__(self, tokenizer):
        self._pad_id = tokenizer.pad_token_id

    def __call__(self, features):
        def pad_group(key_ids, key_mask):
            seqs = [f[key_ids] for f in features]
            masks = [f[key_mask] for f in features]
            max_len = max(s.size(0) for s in seqs)
            padded_ids  = torch.full((len(seqs), max_len), self._pad_id, dtype=torch.long)
            padded_mask = torch.zeros((len(seqs), max_len), dtype=torch.long)
            for i, (s, m) in enumerate(zip(seqs, masks)):
                padded_ids[i, :s.size(0)]  = s
                padded_mask[i, :m.size(0)] = m
            return padded_ids, padded_mask

        a_ids, a_mask = pad_group("a_input_ids", "a_attention_mask")
        b_ids, b_mask = pad_group("b_input_ids", "b_attention_mask")
        return {
            "a_input_ids": a_ids, "a_attention_mask": a_mask,
            "b_input_ids": b_ids, "b_attention_mask": b_mask,
        }


class BTDataset(Dataset):
    def __init__(self, records):
        self.records = records
    def __len__(self):
        return len(self.records)
    def __getitem__(self, idx):
        return dict(self.records[idx])


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_probs(model, loader, device):
    model.eval()
    all_logits = []
    for batch in tqdm(loader, desc="Inference", leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else __import__("contextlib").nullcontext():
            logits = model(
                a_input_ids=batch["a_input_ids"],
                a_attention_mask=batch["a_attention_mask"],
                b_input_ids=batch["b_input_ids"],
                b_attention_mask=batch["b_attention_mask"],
            )
        all_logits.append(logits.cpu().float())
    return F.softmax(torch.cat(all_logits), dim=1).numpy()


def main():
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    print(f"Test rows: {len(test_df)}")

    tokenizer = RobertaTokenizer(
        vocab_file=os.path.join(MODEL_DIR, "vocab.json"),
        merges_file=os.path.join(MODEL_DIR, "merges.txt"),
    )
    collator = BTCollator(tokenizer)

    # Pre-tokenize test
    records = []
    prompts     = test_df["prompt"].apply(lambda s: clean_text(parse_prompt(s))).tolist()
    responses_a = test_df["response_a"].apply(lambda s: clean_text(parse_prompt(s))).tolist()
    responses_b = test_df["response_b"].apply(lambda s: clean_text(parse_prompt(s))).tolist()
    for i in tqdm(range(len(test_df)), desc="Tokenizing"):
        enc_a = encode_pair(tokenizer, prompts[i], responses_a[i], MAX_LEN)
        enc_b = encode_pair(tokenizer, prompts[i], responses_b[i], MAX_LEN)
        records.append({
            "a_input_ids": enc_a["input_ids"], "a_attention_mask": enc_a["attention_mask"],
            "b_input_ids": enc_b["input_ids"], "b_attention_mask": enc_b["attention_mask"],
        })

    loader = DataLoader(BTDataset(records), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator)

    ckpt_paths = sorted(glob.glob(CKPT_PATTERN))
    if not ckpt_paths:
        raise FileNotFoundError(f"No seed checkpoints found at {CKPT_PATTERN}")
    print(f"Found {len(ckpt_paths)} seed checkpoint(s): {[os.path.basename(p) for p in ckpt_paths]}")

    all_preds = []
    for ckpt_path in ckpt_paths:
        print(f"Loading {os.path.basename(ckpt_path)} ...")
        bundle = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        m = BTRewardModel(MODEL_DIR).to(DEVICE)
        m.load_state_dict(bundle["state_dict"])
        all_preds.append(predict_probs(m, loader, DEVICE))
        del m
        torch.cuda.empty_cache()

    test_preds = np.mean(all_preds, axis=0)

    sub = pd.DataFrame(test_preds, columns=LABEL_COLS)
    sub.insert(0, "id", test_df["id"].values)
    sub.to_csv(OUTPUT, index=False)
    print(f"\nSubmission saved to {OUTPUT}")
    print(sub.head())
    assert (sub[LABEL_COLS].sum(axis=1) - 1.0).abs().max() < 1e-4, "Probabilities don't sum to 1!"
    print("Probability sum check passed.")


if __name__ == "__main__":
    main()
