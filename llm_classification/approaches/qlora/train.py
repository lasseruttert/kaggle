"""
train.py — LoRA Qwen2.5-7B 3-Class Classifier
Fine-tunes Qwen2.5-7B with LoRA adapters for sequence classification.
No quantization needed — LoRA freezes base weights so only ~60MB needs gradients.

Memory budget (5070 Ti 16GB):
  7B weights bfloat16 (frozen)  : ~14.0 GB
  LoRA adapters + AdamW states  : ~60 MB
  Activations (grad checkpoint)  : ~1.5 GB
  Score head + grads            : ~30 MB
  Total                         : ~15.6 GB

If OOM: reduce MAX_LEN to 1536 or 1024.
"""

import json
import os
import re
import unicodedata
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
transformers.logging.set_verbosity_error()
from peft import LoraConfig, TaskType, get_peft_model
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_MODEL      = "Qwen/Qwen2.5-7B"
MAX_LEN         = 2048
FOLDS           = 2
EPOCHS          = 2
LR              = 2e-4
WARMUP_RATIO    = 0.05
BATCH_SIZE      = 1       # keep at 1 for 16GB; 7B weights leave ~2GB headroom
GRAD_ACCUM      = 32      # effective batch = 32
LABEL_SMOOTHING = 0.05

_HERE    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE, "..", "..", "llm-classification-finetuning")
CKPT_DIR = os.path.join(_HERE, "..", "..", "checkpoints")

torch.set_float32_matmul_precision("high")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(
    f"CUDA available: {torch.cuda.is_available()}"
    + (f"  |  GPU: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "  |  Running on CPU")
)
print(f"Device: {DEVICE}  |  Base model: {BASE_MODEL}")

LABEL_COLS = ["winner_model_a", "winner_model_b", "winner_tie"]

LORA_CONFIG = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    task_type=TaskType.SEQ_CLS,
)


# ---------------------------------------------------------------------------
# Helpers (verbatim from stacked/train.py)
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


def swap_ab(df: pd.DataFrame) -> pd.DataFrame:
    swapped = df.copy()
    swapped["response_a"]     = df["response_b"]
    swapped["response_b"]     = df["response_a"]
    swapped["winner_model_a"] = df["winner_model_b"]
    swapped["winner_model_b"] = df["winner_model_a"]
    swapped["label"]          = swapped[LABEL_COLS].values.argmax(axis=1)
    return swapped


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_prompt(prompt: str, resp_a: str, resp_b: str) -> str:
    # Plain text — tokenizer adds BOS/EOS via add_special_tokens=True
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
    def __init__(self, df: pd.DataFrame, tokenizer, has_labels: bool = True):
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
            item = {
                "input_ids":      torch.tensor(enc["input_ids"],      dtype=torch.long),
                "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            }
            if has_labels:
                item["labels"] = torch.tensor(
                    int(np.argmax(row[LABEL_COLS].values.astype(float))), dtype=torch.long
                )
            self.records.append(item)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return dict(self.records[idx])


def make_collate_fn(pad_token_id: int):
    """Left-pad to max length in batch (decoder classification on last real token)."""
    def collate(batch: list) -> dict:
        max_len    = max(item["input_ids"].size(0) for item in batch)
        has_labels = "labels" in batch[0]
        input_ids_list, attn_mask_list, labels_list = [], [], []
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
            if has_labels:
                labels_list.append(item["labels"])
        out = {
            "input_ids":      torch.stack(input_ids_list),
            "attention_mask": torch.stack(attn_mask_list),
        }
        if has_labels:
            out["labels"] = torch.stack(labels_list)
        return out
    return collate


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def load_model(base_model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=True)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_path,
        num_labels=3,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
    )
    model.config.pad_token_id = tokenizer.eos_token_id

    model = get_peft_model(model, LORA_CONFIG)
    # Required: allows gradients to flow into frozen base model for gradient checkpointing
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.print_trainable_parameters()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Train / Evaluate
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, scheduler, grad_accum: int, epoch: int, total: int):
    model.train()
    total_loss, n_steps, skipped = 0.0, 0, 0
    optimizer.zero_grad()
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total} [train]", leave=False)
    for step, batch in enumerate(pbar):
        labels = batch.pop("labels").to(DEVICE)
        batch  = {k: v.to(DEVICE) for k, v in batch.items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(**batch)
            loss    = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)(outputs.logits, labels)
            loss    = loss / grad_accum

        if torch.isnan(loss):
            skipped += 1
            optimizer.zero_grad()
            continue

        loss.backward()

        if (step + 1) % grad_accum == 0:
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum
        n_steps    += 1
        pbar.set_postfix(loss=f"{loss.item() * grad_accum:.4f}")

    # Flush remaining gradients at end of epoch
    if len(loader) % grad_accum != 0:
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    if skipped:
        print(f"  [!] Skipped {skipped} NaN loss steps")
    return total_loss / max(n_steps, 1)


@torch.no_grad()
def evaluate(model, loader, labels_np: np.ndarray) -> float:
    model.eval()
    all_logits = []
    for batch in tqdm(loader, desc="Eval", leave=False):
        batch.pop("labels", None)
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(**batch)
        all_logits.append(outputs.logits.cpu().float())
    preds = torch.cat(all_logits).softmax(-1).numpy()
    return log_loss(labels_np, preds, labels=[0, 1, 2])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    train_df["label"] = train_df[LABEL_COLS].values.argmax(axis=1)
    print(f"Train: {len(train_df)}")

    os.makedirs(CKPT_DIR, exist_ok=True)

    kf     = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)
    splits = list(kf.split(train_df, train_df["label"]))

    for fold in range(FOLDS):
        print(f"\n{'='*60}\nFOLD {fold+1}/{FOLDS}\n{'='*60}")
        tr_idx, va_idx = splits[fold]
        tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
        va_df = train_df.iloc[va_idx].reset_index(drop=True)

        tr_df_aug = pd.concat([tr_df, swap_ab(tr_df)], ignore_index=True)
        print(f"  Train: {len(tr_df_aug)} (incl. swap aug)  Val: {len(va_df)}")

        model, tokenizer = load_model(BASE_MODEL)
        collate_fn = make_collate_fn(tokenizer.pad_token_id)

        tr_ds = QLoraDataset(tr_df_aug, tokenizer)
        va_ds = QLoraDataset(va_df, tokenizer)

        tr_loader = DataLoader(
            tr_ds, batch_size=BATCH_SIZE, shuffle=True,
            pin_memory=True, collate_fn=collate_fn,
        )
        va_loader = DataLoader(
            va_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
            pin_memory=True, collate_fn=collate_fn,
        )

        total_steps  = max(1, (len(tr_loader) // GRAD_ACCUM) * EPOCHS)
        warmup_steps = int(WARMUP_RATIO * total_steps)
        optimizer    = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
        scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

        lora_ckpt  = os.path.join(CKPT_DIR, f"qlora_lora_f{fold}")
        score_ckpt = os.path.join(CKPT_DIR, f"best_qlora_score_f{fold}.pt")
        va_labels  = va_df["label"].values
        best_loss  = float("inf")

        for epoch in range(1, EPOCHS + 1):
            train_loss = train_epoch(
                model, tr_loader, optimizer, scheduler, GRAD_ACCUM, epoch, EPOCHS
            )
            val_loss = evaluate(model, va_loader, va_labels)
            print(f"  Epoch {epoch}/{EPOCHS}  train_loss={train_loss:.4f}  val_log_loss={val_loss:.4f}")
            if val_loss < best_loss:
                best_loss = val_loss
                model.save_pretrained(lora_ckpt)
                torch.save(model.base_model.model.score.state_dict(), score_ckpt)
                print(f"    ↳ Saved checkpoint (val_log_loss={best_loss:.4f})")

        del model, tokenizer, tr_ds, va_ds
        torch.cuda.empty_cache()

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
