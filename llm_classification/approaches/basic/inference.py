"""
inference_basic.py — Load fine-tuned RoBERTa fold checkpoints and produce submission.
Designed to run on Kaggle with no internet access.

Expected Kaggle dataset layout:
  /kaggle/input/llm-bert-basic-finetuned/
      best_basic_f0.pt ... best_basic_f4.pt
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
from transformers import AutoTokenizer, AutoModel, AutoConfig
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
KAGGLE       = os.path.exists("/kaggle")
MODEL_DIR    = "/kaggle/input/llm-bert-basic-finetuned" if KAGGLE else "G:/My Drive/kaggle/llm_classification/kaggle_dataset/bert-finetuned"
DATA_DIR     = "/kaggle/input/competitions/llm-classification-finetuning" if KAGGLE else "G:/My Drive/kaggle/llm_classification/llm-classification-finetuning"
OUTPUT       = "/kaggle/working/submission.csv" if KAGGLE else "submission.csv"
MAX_LEN      = 512
BATCH_SIZE   = 16
N_FEATURES   = 8
CKPT_PATTERN = os.path.join(MODEL_DIR, "best_basic_f*.pt")

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


def truncate_parts(tokenizer, prompt, resp_a, resp_b, max_len):
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
    return (
        tokenizer.decode(p_ids, skip_special_tokens=True),
        tokenizer.decode(a_ids, skip_special_tokens=True),
        tokenizer.decode(b_ids, skip_special_tokens=True),
    )


def build_hand_features(resp_a: str, resp_b: str) -> np.ndarray:
    def word_count(t):   return len(t.split())
    def sent_count(t):   return max(1, len(re.split(r"[.!?]+", t)))
    def code_blocks(t):  return t.count("```")
    def md_elements(t):  return len(re.findall(r"^#{1,6}\s|^[-*]\s|^\d+\.", t, re.M))
    def avg_word_len(t): words = t.split(); return np.mean([len(w) for w in words]) if words else 0
    def ttr(t):          words = t.lower().split(); return len(set(words)) / max(len(words), 1)

    la, lb = len(resp_a), len(resp_b)
    total  = la + lb + 1e-9
    wa, wb = word_count(resp_a), word_count(resp_b)

    return np.array([
        la / total,
        float(la > lb),
        (wa - wb) / (wa + wb + 1e-9),
        (sent_count(resp_a) - sent_count(resp_b)) / 10,
        float(code_blocks(resp_a) > 0) - float(code_blocks(resp_b) > 0),
        (md_elements(resp_a) - md_elements(resp_b)) / 5,
        avg_word_len(resp_a) - avg_word_len(resp_b),
        ttr(resp_a) - ttr(resp_b),
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# Custom model (must match train.py exactly)
# ---------------------------------------------------------------------------

class RobertaWithFeatures(nn.Module):
    def __init__(self, model_dir: str):
        super().__init__()
        config = AutoConfig.from_pretrained(model_dir)
        self.backbone = AutoModel.from_config(config)
        self.head = nn.Sequential(
            nn.Linear(config.hidden_size + N_FEATURES, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 3),
        )

    def forward(self, input_ids, attention_mask, hand_features):
        cls = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        logits = self.head(torch.cat([cls, hand_features], dim=-1))
        return logits


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PreferenceDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        prompt     = clean_text(parse_prompt(row["prompt"]))
        resp_a_raw = clean_text(parse_prompt(row["response_a"]))
        resp_b_raw = clean_text(parse_prompt(row["response_b"]))
        prompt, resp_a, resp_b = truncate_parts(self.tokenizer, prompt, resp_a_raw, resp_b_raw, MAX_LEN)
        enc = self.tokenizer(
            prompt,
            f"Response A: {resp_a} Response B: {resp_b}",
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items() if k != "token_type_ids"}
        item["hand_features"] = torch.tensor(build_hand_features(resp_a_raw, resp_b_raw))
        return item


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_logits = []
    for batch in tqdm(loader, desc="Inference", leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        hand_features = batch.pop("hand_features")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                hand_features=hand_features,
            )
        all_logits.append(logits.cpu().float())
    return torch.cat(all_logits).softmax(-1).numpy()


def main():
    test_df   = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    print(f"Test rows: {len(test_df)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    loader    = DataLoader(PreferenceDataset(test_df, tokenizer), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    ckpt_paths = sorted(glob.glob(CKPT_PATTERN))
    if not ckpt_paths:
        raise FileNotFoundError(f"No fold checkpoints found at {CKPT_PATTERN}")
    print(f"Found {len(ckpt_paths)} fold checkpoint(s): {[os.path.basename(p) for p in ckpt_paths]}")

    all_preds = []
    for ckpt_path in ckpt_paths:
        print(f"Loading {os.path.basename(ckpt_path)} ...")
        m = RobertaWithFeatures(MODEL_DIR).to(DEVICE)
        m.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        all_preds.append(predict(m, loader, DEVICE))
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
