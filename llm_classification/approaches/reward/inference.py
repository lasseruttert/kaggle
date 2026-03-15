"""
inference_reward.py — Inference with fine-tuned reward model (RewardFor3Class).
Designed to run on Kaggle with no internet access.

Expected Kaggle dataset layout:
  /kaggle/input/llm-reward-finetuned/
      config.json
      model.safetensors (or pytorch_model.bin)
      tokenizer_config.json
      ...
"""

import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "sentencepiece", "-q"], check=True)

import json
import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
KAGGLE     = os.path.exists("/kaggle")
MODEL_DIR  = "/kaggle/input/llm-reward-finetuned" if KAGGLE else "G:/My Drive/kaggle/llm_classification/kaggle_dataset/reward-finetuned"
DATA_DIR   = "/kaggle/input/competitions/llm-classification-finetuning" if KAGGLE else "G:/My Drive/kaggle/llm_classification/llm-classification-finetuning"
BASE_MODEL = "OpenAssistant/reward-model-deberta-v3-large-v2"
OUTPUT     = "/kaggle/working/submission.csv" if KAGGLE else "submission_reward.csv"
MAX_LEN    = 512
BATCH_SIZE = 8

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"CUDA available: {torch.cuda.is_available()}" + (f"  |  GPU: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "  |  Running on CPU"))
print(f"Device: {DEVICE}  |  Model: {MODEL_DIR}")

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


# ---------------------------------------------------------------------------
# Model (must match train_reward_model.py)
# ---------------------------------------------------------------------------

class RewardFor3Class(nn.Module):
    def __init__(self, base_model_name: str):
        super().__init__()
        config = AutoConfig.from_pretrained(base_model_name)
        self.backbone = AutoModelForSequenceClassification.from_pretrained(base_model_name, dtype=torch.float32)
        self.backbone.classifier = nn.Linear(config.hidden_size, 3)

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        kwargs = dict(input_ids=input_ids, attention_mask=attention_mask)
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        logits = self.backbone(**kwargs).logits
        loss = nn.CrossEntropyLoss()(logits, labels) if labels is not None else None
        return loss, logits


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PreferenceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        prompt = parse_prompt(row["prompt"])
        resp_a = parse_prompt(row["response_a"])
        resp_b = parse_prompt(row["response_b"])
        prompt, resp_a, resp_b = truncate_parts(self.tokenizer, prompt, resp_a, resp_b, MAX_LEN)

        enc = self.tokenizer(
            prompt,
            f"Response A: {resp_a} Response B: {resp_b}",
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in enc.items()}


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_logits = []
    for batch in tqdm(loader, desc="Inference"):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)
        _, logits = model(input_ids, attention_mask, token_type_ids)
        all_logits.append(logits.cpu().float())
    return torch.cat(all_logits).softmax(-1).numpy()


def main():
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    print(f"Test rows: {len(test_df)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model     = RewardFor3Class(BASE_MODEL).to(DEVICE)
    # Load fine-tuned weights saved via save_pretrained (state dict stored in model.safetensors)
    from transformers import AutoModel
    import safetensors.torch as st
    weights_path = os.path.join(MODEL_DIR, "model.safetensors")
    if os.path.exists(weights_path):
        model.load_state_dict(st.load_file(weights_path), strict=False)
    else:
        bin_path = os.path.join(MODEL_DIR, "pytorch_model.bin")
        model.load_state_dict(torch.load(bin_path, map_location="cpu"), strict=False)
    print("Model loaded.")

    loader     = DataLoader(PreferenceDataset(test_df, tokenizer), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_preds = predict(model, loader, DEVICE)

    sub = pd.DataFrame(test_preds, columns=LABEL_COLS)
    sub.insert(0, "id", test_df["id"].values)
    sub.to_csv(OUTPUT, index=False)
    print(f"\nSubmission saved to {OUTPUT}")
    print(sub.head())
    assert (sub[LABEL_COLS].sum(axis=1) - 1.0).abs().max() < 1e-4, "Probabilities don't sum to 1!"
    print("Probability sum check passed.")


if __name__ == "__main__":
    main()
