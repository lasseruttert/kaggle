"""
train.py — RoBERTa Bradley-Terry (Davidson) Reward Model
Each (prompt, response) pair is scored independently; winner is higher score.
Tie modeled via learned tie logit (Davidson model) with cross-entropy loss.
Multiple seeds, no folds — simpler and more robust.
"""

import contextlib
import json
import os
import re
import unicodedata
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import (
    AutoTokenizer, AutoModel,
    get_linear_schedule_with_warmup,
)
transformers.logging.set_verbosity_error()
from torch.optim import AdamW
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL           = "roberta-base"
MAX_LEN         = 512
BATCH_SIZE      = 16
EPOCHS          = 4
LR              = 2e-5
HEAD_LR_MULT    = 10
PROMPT_BUDGET   = 128
SEEDS           = [42, 1337, 2024]
VAL_SIZE        = 0.1
LABEL_SMOOTHING = 0.1
DROPOUT         = 0.1

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


def compute_log_loss(labels: np.ndarray, preds: np.ndarray) -> float:
    return log_loss(labels, preds, labels=[0, 1, 2])


# ---------------------------------------------------------------------------
# Tokenization: works directly with token IDs (no decode→re-encode)
# ---------------------------------------------------------------------------

def encode_pair(tokenizer, prompt: str, response: str, max_len: int) -> dict:
    budget = max_len - 3  # [CLS] prompt [SEP] response [SEP]
    prompt_budget = min(PROMPT_BUDGET, budget // 3)
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
# Davidson Bradley-Terry model (learned tie logit)
# ---------------------------------------------------------------------------

class BTRewardModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            MODEL, attn_implementation="sdpa", torch_dtype=torch.float32
        )
        hidden_size = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(DROPOUT)
        self.score_head = nn.Linear(hidden_size, 1, bias=False)
        # Input-dependent tie head: uses |cls_a - cls_b| (naturally symmetric)
        self.tie_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(128, 1),
        )

    def _encode(self, input_ids, attention_mask):
        cls = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        return self.dropout(cls)

    def forward(self, a_input_ids, a_attention_mask, b_input_ids, b_attention_mask, labels=None):
        cls_a = self._encode(a_input_ids, a_attention_mask)
        cls_b = self._encode(b_input_ids, b_attention_mask)
        score_a = self.score_head(cls_a).squeeze(-1)
        score_b = self.score_head(cls_b).squeeze(-1)
        tie_logit = self.tie_head(torch.abs(cls_a - cls_b)).squeeze(-1)
        logits = torch.stack([
            score_a - score_b,
            score_b - score_a,
            tie_logit,
        ], dim=1)  # [N, 3]
        loss = F.cross_entropy(logits, labels, label_smoothing=LABEL_SMOOTHING) if labels is not None else None
        return loss, logits


# ---------------------------------------------------------------------------
# Dataset / Collator
# ---------------------------------------------------------------------------

def pretokenize_bt(df: pd.DataFrame, tokenizer, has_labels: bool = True) -> list:
    records = []
    prompts     = df["prompt"].apply(lambda s: clean_text(parse_prompt(s))).tolist()
    responses_a = df["response_a"].apply(lambda s: clean_text(parse_prompt(s))).tolist()
    responses_b = df["response_b"].apply(lambda s: clean_text(parse_prompt(s))).tolist()

    if has_labels:
        labels = df[LABEL_COLS].values.argmax(axis=1)

    for i in tqdm(range(len(df)), desc="Tokenizing", leave=False):
        enc_a = encode_pair(tokenizer, prompts[i], responses_a[i], MAX_LEN)
        enc_b = encode_pair(tokenizer, prompts[i], responses_b[i], MAX_LEN)
        item = {
            "a_input_ids":      enc_a["input_ids"],
            "a_attention_mask": enc_a["attention_mask"],
            "b_input_ids":      enc_b["input_ids"],
            "b_attention_mask": enc_b["attention_mask"],
        }
        if has_labels:
            item["labels"] = torch.tensor(int(labels[i]), dtype=torch.long)
        records.append(item)
    return records


def swap_records(records: list) -> list:
    """Swap a/b fields for augmentation. Labels: 0↔1, 2→2."""
    LABEL_SWAP = {0: 1, 1: 0, 2: 2}
    swapped = []
    for r in records:
        s = {
            "a_input_ids":      r["b_input_ids"],
            "a_attention_mask": r["b_attention_mask"],
            "b_input_ids":      r["a_input_ids"],
            "b_attention_mask": r["a_attention_mask"],
        }
        if "labels" in r:
            s["labels"] = torch.tensor(LABEL_SWAP[r["labels"].item()], dtype=torch.long)
        swapped.append(s)
    return swapped


class BTDataset(Dataset):
    def __init__(self, records: list):
        self.records = records
    def __len__(self):
        return len(self.records)
    def __getitem__(self, idx):
        return dict(self.records[idx])


class BTCollator:
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
            "a_input_ids": a_ids, "a_attention_mask": a_mask,
            "b_input_ids": b_ids, "b_attention_mask": b_mask,
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
            loss, _ = model(
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
def predict_probs(model, loader, device, desc="Predicting"):
    """Returns [N, 3] probability array from softmax over Davidson logits."""
    model.eval()
    all_logits = []
    use_cuda = device.type == "cuda"
    for batch in tqdm(loader, desc=desc, leave=False):
        batch.pop("labels", None)
        batch = {k: v.to(device) for k, v in batch.items()}
        ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_cuda else contextlib.nullcontext()
        with ctx:
            _, logits = model(
                a_input_ids=batch["a_input_ids"],
                a_attention_mask=batch["a_attention_mask"],
                b_input_ids=batch["b_input_ids"],
                b_attention_mask=batch["b_attention_mask"],
            )
        all_logits.append(logits.cpu().float())
    return F.softmax(torch.cat(all_logits), dim=1).numpy()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test_df  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

    train_df["label"] = train_df[LABEL_COLS].values.argmax(axis=1)
    print(f"Train: {len(train_df)}  Test: {len(test_df)}  Seeds: {SEEDS}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    collator  = BTCollator(tokenizer)

    # Pre-tokenize everything ONCE
    print("Pre-tokenizing train...")
    all_train_records = pretokenize_bt(train_df, tokenizer, has_labels=True)
    print("Pre-tokenizing test...")
    test_records = pretokenize_bt(test_df, tokenizer, has_labels=False)
    test_loader  = DataLoader(
        BTDataset(test_records), batch_size=BATCH_SIZE * 2,
        shuffle=False, pin_memory=True, collate_fn=collator,
    )

    test_preds = np.zeros((len(test_df), 3), dtype=np.float32)
    os.makedirs(CKPT_DIR, exist_ok=True)

    for seed_idx, seed in enumerate(SEEDS):
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Train/val split with this seed
        all_idx = np.arange(len(train_df))
        tr_idx, va_idx = train_test_split(
            all_idx, test_size=VAL_SIZE, stratify=train_df["label"], random_state=seed
        )

        # Augment train only
        tr_records_orig = [all_train_records[i] for i in tr_idx]
        train_records   = tr_records_orig + swap_records(tr_records_orig)
        val_records     = [all_train_records[i] for i in va_idx]

        print(f"\n{'='*60}")
        print(f"SEED {seed} ({seed_idx+1}/{len(SEEDS)})  train={len(train_records)} (aug)  val={len(val_records)}")
        print(f"{'='*60}")

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
        backbone_params = list(model.backbone.parameters())
        head_params = list(model.score_head.parameters()) + list(model.tie_head.parameters())
        optimizer = AdamW([
            {"params": backbone_params, "lr": LR, "weight_decay": 0.01},
            {"params": head_params, "lr": LR * HEAD_LR_MULT, "weight_decay": 0.01},
        ])
        total_steps = len(train_loader) * EPOCHS
        scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)

        va_labels = train_df["label"].values[va_idx]
        ckpt_path = os.path.join(CKPT_DIR, f"best_bt_s{seed}.pt")
        best_val_loss = float("inf")

        for epoch in range(1, EPOCHS + 1):
            train_loss = train_epoch(model, train_loader, optimizer, scheduler, scaler, DEVICE, epoch, EPOCHS)
            val_preds  = predict_probs(model, val_loader, DEVICE, desc=f"Epoch {epoch}/{EPOCHS} [val]")
            val_loss   = compute_log_loss(va_labels, val_preds)
            val_acc    = accuracy_score(va_labels, val_preds.argmax(axis=1))
            print(f"  Epoch {epoch}/{EPOCHS}  train_loss={train_loss:.4f}  val_log_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({"state_dict": model.state_dict()}, ckpt_path)
                print(f"    ↳ Saved checkpoint (val_log_loss={best_val_loss:.4f})")

        # Load best and predict test
        bundle = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(bundle["state_dict"])

        test_probs = predict_probs(model, test_loader, DEVICE, desc=f"Seed {seed} test")
        test_preds += test_probs / len(SEEDS)

        # Report final val
        val_preds = predict_probs(model, val_loader, DEVICE, desc=f"Seed {seed} final val")
        val_loss  = compute_log_loss(va_labels, val_preds)
        print(f"  Best val_log_loss={val_loss:.4f}")

        del model
        torch.cuda.empty_cache()

    sub = pd.DataFrame(test_preds, columns=LABEL_COLS)
    sub.insert(0, "id", test_df["id"].values)
    sub.to_csv(OUTPUT, index=False)
    print(f"\nSubmission saved to {OUTPUT}")
    print(sub.head())
    assert (sub[LABEL_COLS].sum(axis=1) - 1.0).abs().max() < 1e-4, "Probabilities don't sum to 1!"
    print("Probability sum check passed.")


if __name__ == "__main__":
    main()
