"""
train.py — RoBERTa Bradley-Terry Reward Model
Each (prompt, response) pair is scored independently; winner is higher score.
Tie calibrated post-hoc via threshold on |score_a - score_b|.
"""

import contextlib
import json
import os
import re
import unicodedata
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, accuracy_score
import torch
import torch.nn.functional as F
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
MODEL         = "roberta-base"
MAX_LEN       = 512
BATCH_SIZE    = 16
FOLDS         = 2
EPOCHS        = 4
LR            = 2e-5
PROMPT_BUDGET = 128

_HERE    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE, "..", "..", "llm-classification-finetuning")
CKPT_DIR = os.path.join(_HERE, "..", "..", "checkpoints")
OUTPUT   = "submission.csv"

torch.set_float32_matmul_precision("high")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"CUDA available: {torch.cuda.is_available()}" + (f"  |  GPU: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "  |  Running on CPU"))
print(f"Device: {DEVICE}  |  Model: {MODEL}")

LABEL_COLS = ["winner_model_a", "winner_model_b", "winner_tie"]


# ---------------------------------------------------------------------------
# Helpers (verbatim from basic/train.py)
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
# BT-specific truncation: single (prompt, response) pair
# ---------------------------------------------------------------------------

def truncate_pair(tokenizer, prompt: str, response: str, max_len: int) -> tuple[str, str]:
    """Truncate prompt+response to fit in max_len tokens for a single pair input."""
    budget = max_len - 3  # [CLS] prompt [SEP] response [SEP]
    prompt_budget = min(PROMPT_BUDGET, budget // 3)
    response_budget = budget - prompt_budget

    p_ids = tokenizer.encode(prompt,   add_special_tokens=False)
    r_ids = tokenizer.encode(response, add_special_tokens=False)

    p_ids = p_ids[:prompt_budget]
    r_ids = r_ids[:response_budget]

    return (
        tokenizer.decode(p_ids, skip_special_tokens=True),
        tokenizer.decode(r_ids, skip_special_tokens=True),
    )


# ---------------------------------------------------------------------------
# Bradley-Terry model
# ---------------------------------------------------------------------------

class BTRewardModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            MODEL, attn_implementation="sdpa", torch_dtype=torch.float32
        )
        hidden_size = self.backbone.config.hidden_size
        self.score_head = nn.Linear(hidden_size, 1, bias=False)

    def score(self, input_ids, attention_mask):
        cls = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        return self.score_head(cls).squeeze(-1)

    def forward(self, a_input_ids, a_attention_mask, b_input_ids, b_attention_mask, labels=None):
        score_a = self.score(a_input_ids, a_attention_mask)
        score_b = self.score(b_input_ids, b_attention_mask)
        loss = bt_loss(score_a, score_b, labels) if labels is not None else None
        return loss, score_a, score_b


# ---------------------------------------------------------------------------
# Bradley-Terry loss
# ---------------------------------------------------------------------------

def bt_loss(score_a, score_b, labels):
    """
    labels: 0 = A wins, 1 = B wins, 2 = tie
    """
    diff = score_a - score_b
    loss_a_wins = F.softplus(-diff)                         # label 0
    loss_b_wins = F.softplus(diff)                          # label 1
    loss_tie    = 0.5 * (loss_a_wins + loss_b_wins)         # label 2 — symmetric

    mask_a = (labels == 0).float()
    mask_b = (labels == 1).float()
    mask_tie = (labels == 2).float()

    return (mask_a * loss_a_wins + mask_b * loss_b_wins + mask_tie * loss_tie).mean()


# ---------------------------------------------------------------------------
# Score → probability calibration
# ---------------------------------------------------------------------------

def scores_to_probs(score_a, score_b, threshold: float, temp: float) -> np.ndarray:
    """
    Convert scalar scores to [N, 3] probability array [p_a, p_b, p_tie].
    score_a, score_b: np.ndarray [N]
    Returns np.ndarray [N, 3]
    """
    diff = score_a - score_b
    raw_a   = 1.0 / (1.0 + np.exp(-(diff - threshold) / temp))
    raw_b   = 1.0 / (1.0 + np.exp(-(-diff - threshold) / temp))
    raw_tie = np.maximum(0.0, 1.0 - raw_a - raw_b)
    total   = raw_a + raw_b + raw_tie + 1e-9
    return np.stack([raw_a / total, raw_b / total, raw_tie / total], axis=1)


def calibrate_threshold(oof_sa: np.ndarray, oof_sb: np.ndarray, oof_labels: np.ndarray):
    """
    Grid search over (threshold, temp) to minimize log-loss on OOF scores.
    Returns (best_threshold, best_temp).
    """
    diffs = np.abs(oof_sa - oof_sb)
    thresholds = np.linspace(0.0, float(np.percentile(diffs, 95)), 101)
    temperatures = [0.25, 0.5, 1.0, 2.0, 4.0]

    best_loss = float("inf")
    best_threshold, best_temp = 0.0, 1.0

    for thresh in thresholds:
        for temp in temperatures:
            probs = scores_to_probs(oof_sa, oof_sb, thresh, temp)
            loss  = log_loss(oof_labels, probs, labels=[0, 1, 2])
            if loss < best_loss:
                best_loss = loss
                best_threshold = float(thresh)
                best_temp = float(temp)

    return best_threshold, best_temp


# ---------------------------------------------------------------------------
# Dataset / Collator
# ---------------------------------------------------------------------------

def pretokenize_bt(df: pd.DataFrame, tokenizer, has_labels: bool = True) -> list:
    records = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Tokenizing", leave=False):
        prompt     = clean_text(parse_prompt(row["prompt"]))
        resp_a_raw = clean_text(parse_prompt(row["response_a"]))
        resp_b_raw = clean_text(parse_prompt(row["response_b"]))

        p_a, r_a = truncate_pair(tokenizer, prompt, resp_a_raw, MAX_LEN)
        p_b, r_b = truncate_pair(tokenizer, prompt, resp_b_raw, MAX_LEN)

        enc_a = tokenizer(p_a, r_a, max_length=MAX_LEN, padding=False, truncation=True)
        enc_b = tokenizer(p_b, r_b, max_length=MAX_LEN, padding=False, truncation=True)

        item = {
            "a_input_ids":      torch.tensor(enc_a["input_ids"]),
            "a_attention_mask": torch.tensor(enc_a["attention_mask"]),
            "b_input_ids":      torch.tensor(enc_b["input_ids"]),
            "b_attention_mask": torch.tensor(enc_b["attention_mask"]),
        }
        if has_labels:
            item["labels"] = torch.tensor(
                int(np.argmax(row[LABEL_COLS].values.astype(float))), dtype=torch.long
            )
        records.append(item)
    return records


class BTDataset(Dataset):
    def __init__(self, records: list):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return dict(self.records[idx])


class BTCollator:
    """Pads a_* and b_* groups independently (they may differ in length)."""
    def __init__(self, tokenizer):
        self._pad_id = tokenizer.pad_token_id

    def __call__(self, features):
        labels = None
        if "labels" in features[0]:
            labels = torch.stack([f["labels"] for f in features])

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

        batch = {
            "a_input_ids":      a_ids,
            "a_attention_mask": a_mask,
            "b_input_ids":      b_ids,
            "b_attention_mask": b_mask,
        }
        if labels is not None:
            batch["labels"] = labels
        return batch


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, scheduler, scaler, device, epoch, total_epochs):
    model.train()
    total_loss, n_steps = 0.0, 0
    use_cuda = device.type == "cuda"
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [train]", leave=False)
    for batch in pbar:
        batch  = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop("labels")
        ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_cuda else contextlib.nullcontext()
        with ctx:
            loss, _, _ = model(
                a_input_ids=batch["a_input_ids"],
                a_attention_mask=batch["a_attention_mask"],
                b_input_ids=batch["b_input_ids"],
                b_attention_mask=batch["b_attention_mask"],
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
def predict_scores(model, loader, device, desc="Predicting"):
    """Returns (score_a [N], score_b [N]) as numpy arrays."""
    model.eval()
    all_sa, all_sb = [], []
    use_cuda = device.type == "cuda"
    for batch in tqdm(loader, desc=desc, leave=False):
        batch.pop("labels", None)
        batch = {k: v.to(device) for k, v in batch.items()}
        ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_cuda else contextlib.nullcontext()
        with ctx:
            _, sa, sb = model(
                a_input_ids=batch["a_input_ids"],
                a_attention_mask=batch["a_attention_mask"],
                b_input_ids=batch["b_input_ids"],
                b_attention_mask=batch["b_attention_mask"],
            )
        all_sa.append(sa.cpu().float())
        all_sb.append(sb.cpu().float())
    return torch.cat(all_sa).numpy(), torch.cat(all_sb).numpy()


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
    collator  = BTCollator(tokenizer)

    test_records = pretokenize_bt(test_df, tokenizer, has_labels=False)
    test_loader  = DataLoader(
        BTDataset(test_records), batch_size=BATCH_SIZE * 2,
        shuffle=False, pin_memory=True, collate_fn=collator,
    )

    if FOLDS >= 2:
        splitter = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)
        splits   = list(splitter.split(aug_df, aug_df["label"]))
    else:
        from sklearn.model_selection import train_test_split as _tts
        tr_idx, va_idx = _tts(np.arange(len(aug_df)), test_size=0.2, stratify=aug_df["label"], random_state=42)
        splits = [(tr_idx, va_idx)]

    oof_sa   = np.zeros(len(aug_df), dtype=np.float32)
    oof_sb   = np.zeros(len(aug_df), dtype=np.float32)
    oof_mask = np.zeros(len(aug_df), dtype=bool)
    test_preds = np.zeros((len(test_df), 3), dtype=np.float32)

    os.makedirs(CKPT_DIR, exist_ok=True)

    for fold, (tr_idx, va_idx) in enumerate(splits):
        print(f"\n{'='*60}")
        print(f"FOLD {fold+1}/{FOLDS}  train={len(tr_idx)}  val={len(va_idx)}")
        print(f"{'='*60}")

        tr_df = aug_df.iloc[tr_idx].reset_index(drop=True)
        va_df = aug_df.iloc[va_idx].reset_index(drop=True)

        train_records = pretokenize_bt(tr_df, tokenizer)
        val_records   = pretokenize_bt(va_df, tokenizer)

        train_loader = DataLoader(
            BTDataset(train_records), batch_size=BATCH_SIZE,
            shuffle=True, pin_memory=True, collate_fn=collator,
        )
        val_loader = DataLoader(
            BTDataset(val_records), batch_size=BATCH_SIZE * 2,
            shuffle=False, pin_memory=True, collate_fn=collator,
        )

        model     = BTRewardModel().to(DEVICE)
        scaler    = torch.amp.GradScaler()
        optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
        total_steps = len(train_loader) * EPOCHS
        scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)

        best_val_loss = float("inf")
        va_labels     = va_df["label"].values
        ckpt_path     = os.path.join(CKPT_DIR, f"best_bt_f{fold}.pt")

        for epoch in range(1, EPOCHS + 1):
            train_loss = train_epoch(model, train_loader, optimizer, scheduler, scaler, DEVICE, epoch, EPOCHS)
            val_sa, val_sb = predict_scores(model, val_loader, DEVICE, desc=f"Epoch {epoch}/{EPOCHS} [val]")
            # Monitor with default threshold=0, temp=1
            val_preds = scores_to_probs(val_sa, val_sb, threshold=0.0, temp=1.0)
            val_loss  = compute_log_loss(va_labels, val_preds)
            val_acc   = accuracy_score(va_labels, val_preds.argmax(axis=1))
            print(f"  Epoch {epoch}/{EPOCHS}  train_loss={train_loss:.4f}  val_log_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Save temp checkpoint (state_dict only; calibration added below)
                torch.save({"state_dict": model.state_dict()}, ckpt_path)
                print(f"    ↳ Saved checkpoint (val_log_loss={best_val_loss:.4f})")

        # Load best, collect OOF scores, calibrate threshold
        bundle = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(bundle["state_dict"])

        oof_va_sa, oof_va_sb = predict_scores(model, val_loader, DEVICE, desc=f"Fold {fold+1} OOF")
        oof_sa[va_idx]   = oof_va_sa
        oof_sb[va_idx]   = oof_va_sb
        oof_mask[va_idx] = True

        best_threshold, best_temp = calibrate_threshold(oof_va_sa, oof_va_sb, va_labels)
        print(f"  Calibrated: threshold={best_threshold:.4f}  temp={best_temp:.4f}")

        calibrated_val_preds = scores_to_probs(oof_va_sa, oof_va_sb, best_threshold, best_temp)
        cal_val_loss = compute_log_loss(va_labels, calibrated_val_preds)
        print(f"  Calibrated val_log_loss={cal_val_loss:.4f}  (vs uncalibrated {best_val_loss:.4f})")

        # Re-save bundle with calibrated values
        torch.save({
            "state_dict": model.state_dict(),
            "threshold":  best_threshold,
            "temperature": best_temp,
        }, ckpt_path)
        print(f"  Saved calibrated bundle → {ckpt_path}")

        # Test predictions for this fold
        test_sa, test_sb = predict_scores(model, test_loader, DEVICE, desc=f"Fold {fold+1} test")
        test_preds += scores_to_probs(test_sa, test_sb, best_threshold, best_temp) / len(splits)

    oof_labels = aug_df["label"].values[oof_mask]
    oof_probs  = scores_to_probs(oof_sa[oof_mask], oof_sb[oof_mask], threshold=0.0, temp=1.0)
    oof_loss   = compute_log_loss(oof_labels, oof_probs)
    print(f"\nOOF log-loss (default threshold): {oof_loss:.4f}")

    sub = pd.DataFrame(test_preds, columns=LABEL_COLS)
    sub.insert(0, "id", test_df["id"].values)
    sub.to_csv(OUTPUT, index=False)
    print(f"Submission saved to {OUTPUT}")
    print(sub.head())
    assert (sub[LABEL_COLS].sum(axis=1) - 1.0).abs().max() < 1e-4, "Probabilities don't sum to 1!"
    print("Probability sum check passed.")


if __name__ == "__main__":
    main()
