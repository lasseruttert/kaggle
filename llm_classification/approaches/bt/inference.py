"""
inference.py — Bradley-Terry RoBERTa inference.
Loads calibrated fold bundles and produces submission.csv.
Designed to run on Kaggle with no internet access.

Expected Kaggle dataset layout:
  /kaggle/input/llm-bert-bt-finetuned/
      best_bt_f0.pt ... best_bt_fN.pt
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
CKPT_PATTERN = os.path.join(MODEL_DIR, "best_bt_f*.pt")

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


def truncate_pair(tokenizer, prompt: str, response: str, max_len: int):
    budget = max_len - 3
    prompt_budget   = min(PROMPT_BUDGET, budget // 3)
    response_budget = budget - prompt_budget
    p_ids = tokenizer.encode(prompt,   add_special_tokens=False)[:prompt_budget]
    r_ids = tokenizer.encode(response, add_special_tokens=False)[:response_budget]
    return (
        tokenizer.decode(p_ids, skip_special_tokens=True),
        tokenizer.decode(r_ids, skip_special_tokens=True),
    )


def scores_to_probs(score_a, score_b, threshold: float, temp: float) -> np.ndarray:
    diff    = score_a - score_b
    raw_a   = 1.0 / (1.0 + np.exp(-(diff - threshold) / temp))
    raw_b   = 1.0 / (1.0 + np.exp(-(-diff - threshold) / temp))
    raw_tie = np.maximum(0.0, 1.0 - raw_a - raw_b)
    total   = raw_a + raw_b + raw_tie + 1e-9
    return np.stack([raw_a / total, raw_b / total, raw_tie / total], axis=1)


# ---------------------------------------------------------------------------
# Model (must match train.py exactly)
# ---------------------------------------------------------------------------

class BTRewardModel(nn.Module):
    def __init__(self, model_dir: str):
        super().__init__()
        config = RobertaConfig.from_json_file(os.path.join(model_dir, "config.json"))
        self.backbone   = RobertaModel(config)
        self.score_head = nn.Linear(config.hidden_size, 1, bias=False)

    def score(self, input_ids, attention_mask):
        cls = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        return self.score_head(cls).squeeze(-1)

    def forward(self, a_input_ids, a_attention_mask, b_input_ids, b_attention_mask):
        score_a = self.score(a_input_ids, a_attention_mask)
        score_b = self.score(b_input_ids, b_attention_mask)
        return score_a, score_b


# ---------------------------------------------------------------------------
# Dataset / Collator
# ---------------------------------------------------------------------------

class BTDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.df        = df.reset_index(drop=True)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row        = self.df.iloc[idx]
        prompt     = clean_text(parse_prompt(row["prompt"]))
        resp_a_raw = clean_text(parse_prompt(row["response_a"]))
        resp_b_raw = clean_text(parse_prompt(row["response_b"]))

        p_a, r_a = truncate_pair(self.tokenizer, prompt, resp_a_raw, MAX_LEN)
        p_b, r_b = truncate_pair(self.tokenizer, prompt, resp_b_raw, MAX_LEN)

        enc_a = self.tokenizer(p_a, r_a, max_length=MAX_LEN, padding="max_length", truncation=True, return_tensors="pt")
        enc_b = self.tokenizer(p_b, r_b, max_length=MAX_LEN, padding="max_length", truncation=True, return_tensors="pt")

        return {
            "a_input_ids":      enc_a["input_ids"].squeeze(0),
            "a_attention_mask": enc_a["attention_mask"].squeeze(0),
            "b_input_ids":      enc_b["input_ids"].squeeze(0),
            "b_attention_mask": enc_b["attention_mask"].squeeze(0),
        }


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_scores(model, loader, device):
    model.eval()
    all_sa, all_sb = [], []
    for batch in tqdm(loader, desc="Inference", leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else __import__("contextlib").nullcontext():
            sa, sb = model(
                a_input_ids=batch["a_input_ids"],
                a_attention_mask=batch["a_attention_mask"],
                b_input_ids=batch["b_input_ids"],
                b_attention_mask=batch["b_attention_mask"],
            )
        all_sa.append(sa.cpu().float())
        all_sb.append(sb.cpu().float())
    return torch.cat(all_sa).numpy(), torch.cat(all_sb).numpy()


def main():
    test_df   = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    print(f"Test rows: {len(test_df)}")

    tokenizer = RobertaTokenizer(
        vocab_file=os.path.join(MODEL_DIR, "vocab.json"),
        merges_file=os.path.join(MODEL_DIR, "merges.txt"),
    )
    loader    = DataLoader(BTDataset(test_df, tokenizer), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    ckpt_paths = sorted(glob.glob(CKPT_PATTERN))
    if not ckpt_paths:
        raise FileNotFoundError(f"No fold checkpoints found at {CKPT_PATTERN}")
    print(f"Found {len(ckpt_paths)} fold checkpoint(s): {[os.path.basename(p) for p in ckpt_paths]}")

    all_preds = []
    for ckpt_path in ckpt_paths:
        print(f"Loading {os.path.basename(ckpt_path)} ...")
        bundle = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        threshold   = bundle.get("threshold",   0.0)
        temperature = bundle.get("temperature", 1.0)
        print(f"  threshold={threshold:.4f}  temperature={temperature:.4f}")

        m = BTRewardModel(MODEL_DIR).to(DEVICE)
        m.load_state_dict(bundle["state_dict"])
        sa, sb = predict_scores(m, loader, DEVICE)
        all_preds.append(scores_to_probs(sa, sb, threshold, temperature))
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
