"""
train_basic.py — RoBERTa + Swap Aug + K-Fold + Hand Features
"""

import json
import os
import re
import unicodedata
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, accuracy_score, f1_score
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import (
    AutoTokenizer, AutoModel, AutoConfig,
    get_linear_schedule_with_warmup, DataCollatorWithPadding,
)
transformers.logging.set_verbosity_error()
from torch.optim import AdamW
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL           = "roberta-base"
MAX_LEN         = 512
BATCH_SIZE      = 32
FOLDS           = 1
EPOCHS          = 6
LR              = 2e-5
LABEL_SMOOTHING = 0.1
N_FEATURES      = 8

_HERE     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(_HERE, "..", "..", "llm-classification-finetuning")
CKPT_DIR  = os.path.join(_HERE, "..", "..", "checkpoints")
OUTPUT    = "submission.csv"

torch.set_float32_matmul_precision("high")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"CUDA available: {torch.cuda.is_available()}" + (f"  |  GPU: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "  |  Running on CPU"))
print(f"Device: {DEVICE}  |  Model: {MODEL}")

LABEL_COLS = ["winner_model_a", "winner_model_b", "winner_tie"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_prompt(s: str) -> str:
    """JSON list of strings → newline-joined string."""
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


def compute_log_loss(labels: np.ndarray, preds: np.ndarray) -> float:
    return log_loss(labels, preds, labels=[0, 1, 2])


def truncate_parts(tokenizer, prompt: str, resp_a: str, resp_b: str, max_len: int):
    """Proportionally truncate each segment so the total fits in max_len tokens."""
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
# Swap augmentation
# ---------------------------------------------------------------------------

def swap_ab(df: pd.DataFrame) -> pd.DataFrame:
    swapped = df.copy()
    swapped["response_a"]     = df["response_b"]
    swapped["response_b"]     = df["response_a"]
    swapped["winner_model_a"] = df["winner_model_b"]
    swapped["winner_model_b"] = df["winner_model_a"]
    # winner_tie is symmetric — no change
    swapped["label"] = swapped[LABEL_COLS].values.argmax(axis=1)
    return swapped


# ---------------------------------------------------------------------------
# Custom model
# ---------------------------------------------------------------------------

class RobertaWithFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        config = AutoConfig.from_pretrained(MODEL, num_labels=3)
        self.backbone = AutoModel.from_pretrained(MODEL, attn_implementation="sdpa", torch_dtype=torch.float32)
        self.head = nn.Sequential(
            nn.Linear(config.hidden_size + N_FEATURES, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 3),
        )

    def forward(self, input_ids, attention_mask, hand_features, labels=None):
        cls = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        logits = self.head(torch.cat([cls, hand_features], dim=-1))
        loss = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)(logits, labels) if labels is not None else None
        return loss, logits


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def pretokenize(df: pd.DataFrame, tokenizer, has_labels: bool = True) -> list:
    records = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Tokenizing", leave=False):
        prompt = clean_text(parse_prompt(row["prompt"]))
        resp_a_raw = clean_text(parse_prompt(row["response_a"]))
        resp_b_raw = clean_text(parse_prompt(row["response_b"]))
        prompt, resp_a, resp_b = truncate_parts(tokenizer, prompt, resp_a_raw, resp_b_raw, MAX_LEN)
        enc = tokenizer(
            prompt,
            f"Response A: {resp_a} Response B: {resp_b}",
            max_length=MAX_LEN,
            padding=False,
            truncation=True,
        )
        item = {k: torch.tensor(v) for k, v in enc.items() if k != "token_type_ids"}
        item["hand_features"] = torch.tensor(build_hand_features(resp_a_raw, resp_b_raw))
        if has_labels:
            item["labels"] = torch.tensor(int(np.argmax(row[LABEL_COLS].values.astype(float))), dtype=torch.long)
        records.append(item)
    return records


class PreferenceDataset(Dataset):
    def __init__(self, records: list):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return dict(self.records[idx])


class HandFeatureCollator:
    """DataCollatorWithPadding but also stacks hand_features (fixed-size)."""
    def __init__(self, tokenizer):
        self._pad = DataCollatorWithPadding(tokenizer)

    def __call__(self, features):
        hand_features = torch.stack([f.pop("hand_features") for f in features])
        batch = self._pad(features)
        batch["hand_features"] = hand_features
        return batch


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, scheduler, scaler, device, epoch, total_epochs):
    model.train()
    total_loss, n_steps = 0.0, 0
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [train]", leave=False)
    for batch in pbar:
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop("labels")
        hand_features = batch.pop("hand_features")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss, logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                hand_features=hand_features,
                labels=labels,
            )
        if torch.isnan(loss):
            optimizer.zero_grad()
            continue
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(grad_norm):
            optimizer.zero_grad()
            scaler.update()
            continue
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad()
        total_loss += loss.item()
        n_steps    += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / max(n_steps, 1)


@torch.no_grad()
def predict(model, loader, device, desc="Predicting"):
    model.eval()
    all_logits = []
    for batch in tqdm(loader, desc=desc, leave=False):
        batch.pop("labels", None)
        batch = {k: v.to(device) for k, v in batch.items()}
        hand_features = batch.pop("hand_features")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                hand_features=hand_features,
            )
        all_logits.append(logits.cpu().float())
    return torch.cat(all_logits).softmax(-1).numpy()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test_df  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

    train_df["label"] = train_df[LABEL_COLS].values.argmax(axis=1)

    # Swap augmentation — doubles training data
    aug_df = pd.concat([train_df, swap_ab(train_df)], ignore_index=True)
    print(f"Train (augmented): {len(aug_df)}  Test: {len(test_df)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    collator  = HandFeatureCollator(tokenizer)

    test_records = pretokenize(test_df, tokenizer, has_labels=False)
    test_loader  = DataLoader(PreferenceDataset(test_records), batch_size=BATCH_SIZE * 2, shuffle=False, pin_memory=True, collate_fn=collator)

    if FOLDS >= 2:
        splitter = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)
        splits = list(splitter.split(aug_df, aug_df["label"]))
    else:
        # FOLDS=1: single 80/20 holdout
        from sklearn.model_selection import train_test_split as _tts
        tr_idx, va_idx = _tts(np.arange(len(aug_df)), test_size=0.2, stratify=aug_df["label"], random_state=42)
        splits = [(tr_idx, va_idx)]

    oof_preds  = np.zeros((len(aug_df), 3))
    oof_mask   = np.zeros(len(aug_df), dtype=bool)
    test_preds = np.zeros((len(test_df), 3))

    os.makedirs(CKPT_DIR, exist_ok=True)

    for fold, (tr_idx, va_idx) in enumerate(splits):
        print(f"\n{'='*60}")
        print(f"FOLD {fold+1}/{FOLDS}  train={len(tr_idx)}  val={len(va_idx)}")
        print(f"{'='*60}")

        tr_df = aug_df.iloc[tr_idx].reset_index(drop=True)
        va_df = aug_df.iloc[va_idx].reset_index(drop=True)

        train_records = pretokenize(tr_df, tokenizer)
        val_records   = pretokenize(va_df, tokenizer)

        train_loader = DataLoader(PreferenceDataset(train_records), batch_size=BATCH_SIZE,     shuffle=True,  pin_memory=True, collate_fn=collator)
        val_loader   = DataLoader(PreferenceDataset(val_records),   batch_size=BATCH_SIZE * 2, shuffle=False, pin_memory=True, collate_fn=collator)

        model     = RobertaWithFeatures().to(DEVICE)
        scaler    = torch.amp.GradScaler()
        optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
        total_steps = len(train_loader) * EPOCHS
        scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)

        best_val_loss = float("inf")
        va_labels = va_df["label"].values
        ckpt_path = os.path.join(CKPT_DIR, f"best_basic_f{fold}.pt")

        for epoch in range(1, EPOCHS + 1):
            train_loss = train_epoch(model, train_loader, optimizer, scheduler, scaler, DEVICE, epoch, EPOCHS)
            val_preds  = predict(model, val_loader, DEVICE, desc=f"Epoch {epoch}/{EPOCHS} [val]")
            val_loss   = compute_log_loss(va_labels, val_preds)
            val_acc    = accuracy_score(va_labels, val_preds.argmax(axis=1))
            val_f1     = f1_score(va_labels, val_preds.argmax(axis=1), average="macro")
            print(f"  Epoch {epoch}/{EPOCHS}  train_loss={train_loss:.4f}  val_log_loss={val_loss:.4f}  val_acc={val_acc:.4f}  val_f1={val_f1:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), ckpt_path)
                print(f"    ↳ Saved checkpoint (val_log_loss={best_val_loss:.4f})")

        # Load best and collect OOF + test preds
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        oof_preds[va_idx] = predict(model, val_loader, DEVICE, desc=f"Fold {fold+1} OOF")
        oof_mask[va_idx]  = True
        test_preds        += predict(model, test_loader, DEVICE, desc=f"Fold {fold+1} test") / len(splits)

    oof_loss = compute_log_loss(aug_df["label"].values[oof_mask], oof_preds[oof_mask])
    print(f"\nOOF log-loss: {oof_loss:.4f}")

    sub = pd.DataFrame(test_preds, columns=LABEL_COLS)
    sub.insert(0, "id", test_df["id"].values)
    sub.to_csv(OUTPUT, index=False)
    print(f"Submission saved to {OUTPUT}")
    print(sub.head())
    assert (sub[LABEL_COLS].sum(axis=1) - 1.0).abs().max() < 1e-4, "Probabilities don't sum to 1!"
    print("Probability sum check passed.")


if __name__ == "__main__":
    main()
