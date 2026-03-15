"""
train_reward_model.py — Fine-tune OpenAssistant reward model for 3-class preference

Set INFERENCE_ONLY = True to run zero-shot reward scoring instead of fine-tuning.
"""

import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_MODEL     = "OpenAssistant/reward-model-deberta-v3-large-v2"
FOLDS          = 2
MAX_LEN        = 512
BATCH_SIZE     = 8
EPOCHS         = 3
LR             = 1e-5
INFERENCE_ONLY = False   # True → zero-shot reward scoring, no fine-tuning
_HERE          = os.path.dirname(os.path.abspath(__file__))
DATA_DIR       = os.path.join(_HERE, "..", "..", "llm-classification-finetuning")
CKPT_DIR       = os.path.join(_HERE, "..", "..", "checkpoints")
OUTPUT         = "submission_reward.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"CUDA available: {torch.cuda.is_available()}" + (f"  |  GPU: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "  |  Running on CPU"))
print(f"Device: {DEVICE}  |  Base model: {BASE_MODEL}")

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


def compute_log_loss(labels: np.ndarray, preds: np.ndarray) -> float:
    return log_loss(labels, preds, labels=[0, 1, 2])


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
# Dataset
# ---------------------------------------------------------------------------

class PreferenceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, has_labels: bool = True):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.has_labels = has_labels

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
        item = {k: v.squeeze(0) for k, v in enc.items()}

        if self.has_labels:
            item["labels"] = torch.tensor(int(np.argmax(row[LABEL_COLS].values.astype(float))), dtype=torch.long)
        return item


# ---------------------------------------------------------------------------
# Zero-shot inference using raw reward scores
# ---------------------------------------------------------------------------

def zero_shot_predict(df: pd.DataFrame, tokenizer, device):
    print("Running zero-shot reward scoring ...")
    reward_model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, dtype=torch.float32).eval().to(device)

    all_probs = []
    for i in tqdm(range(0, len(df), BATCH_SIZE), desc="Zero-shot scoring"):
        chunk = df.iloc[i:i + BATCH_SIZE]
        texts_a, texts_b = [], []
        for _, row in chunk.iterrows():
            p = parse_prompt(row["prompt"])
            texts_a.append(f"{p}\n\n{parse_prompt(row['response_a'])}"[:2000])
            texts_b.append(f"{p}\n\n{parse_prompt(row['response_b'])}"[:2000])

        with torch.no_grad():
            s_a = reward_model(**tokenizer(texts_a, max_length=MAX_LEN, padding=True, truncation=True, return_tensors="pt").to(device)).logits.squeeze(-1).cpu().float().numpy()
            s_b = reward_model(**tokenizer(texts_b, max_length=MAX_LEN, padding=True, truncation=True, return_tensors="pt").to(device)).logits.squeeze(-1).cpu().float().numpy()

        for sa, sb in zip(s_a, s_b):
            tie_score = max(0.0, 1.0 - abs(sa - sb))
            raw = np.array([sa, sb, tie_score], dtype=np.float32) - min(sa, sb, tie_score)
            all_probs.append(np.exp(raw) / np.exp(raw).sum())

    return np.array(all_probs)


# ---------------------------------------------------------------------------
# Fine-tune model (3-class head on reward backbone)
# ---------------------------------------------------------------------------

class RewardFor3Class(nn.Module):
    def __init__(self):
        super().__init__()
        config = AutoConfig.from_pretrained(BASE_MODEL)
        self.backbone = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, dtype=torch.float32)
        self.backbone.classifier = nn.Linear(config.hidden_size, 3)

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        kwargs = dict(input_ids=input_ids, attention_mask=attention_mask)
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        logits = self.backbone(**kwargs).logits
        loss = nn.CrossEntropyLoss()(logits, labels) if labels is not None else None
        return loss, logits


def train_epoch(model, loader, optimizer, scheduler, device, epoch, total_epochs, fold_desc):
    model.train()
    total_loss, n_steps = 0.0, 0
    pbar = tqdm(loader, desc=f"{fold_desc} Epoch {epoch}/{total_epochs} [train]", leave=False)
    for batch in pbar:
        batch = {k: v.to(device) for k, v in batch.items()}
        loss, _ = model(batch["input_ids"], batch["attention_mask"], batch.get("token_type_ids"), batch.get("labels"))
        if torch.isnan(loss):
            optimizer.zero_grad()
            continue
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(grad_norm):
            optimizer.zero_grad()
            continue
        optimizer.step()
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
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)
        _, logits = model(input_ids, attention_mask, token_type_ids)
        all_logits.append(logits.cpu().float())
    return torch.cat(all_logits).softmax(-1).numpy()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test_df  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    if INFERENCE_ONLY:
        test_preds = zero_shot_predict(test_df, tokenizer, DEVICE)
        sub = pd.DataFrame(test_preds, columns=LABEL_COLS)
        sub.insert(0, "id", test_df["id"].values)
        sub.to_csv(OUTPUT, index=False)
        print(f"\nZero-shot submission saved to {OUTPUT}")
        print(sub.head())
        return

    train_df["label"] = train_df[LABEL_COLS].values.argmax(axis=1)
    labels_arr = train_df["label"].values
    print(f"Train: {len(train_df)}  Test: {len(test_df)}")

    test_loader = DataLoader(PreferenceDataset(test_df, tokenizer, has_labels=False), batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=0)

    all_val_preds  = np.zeros((len(train_df), 3))
    all_test_preds = np.zeros((len(test_df),  3))

    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(train_df, labels_arr)):
        fold_desc = f"[fold={fold+1}/{FOLDS}]"
        print(f"\nFold {fold+1}/{FOLDS}")

        tr_loader = DataLoader(PreferenceDataset(train_df.iloc[tr_idx], tokenizer), batch_size=BATCH_SIZE,     shuffle=True,  num_workers=0)
        va_loader = DataLoader(PreferenceDataset(train_df.iloc[va_idx], tokenizer), batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=0)

        model        = RewardFor3Class().to(DEVICE)
        optimizer    = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
        total_steps  = len(tr_loader) * EPOCHS
        scheduler    = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)

        best_val_loss = float("inf")
        os.makedirs(CKPT_DIR, exist_ok=True)
        ckpt = os.path.join(CKPT_DIR, f"best_reward_f{fold}.pt")

        for epoch in range(1, EPOCHS + 1):
            tr_loss  = train_epoch(model, tr_loader, optimizer, scheduler, DEVICE, epoch, EPOCHS, fold_desc)
            va_preds = predict(model, va_loader, DEVICE, desc=f"{fold_desc} val")
            va_loss  = compute_log_loss(labels_arr[va_idx], va_preds)
            print(f"  Epoch {epoch}: train_loss={tr_loss:.4f}  val_log_loss={va_loss:.4f}")
            if va_loss < best_val_loss:
                best_val_loss = va_loss
                torch.save(model.state_dict(), ckpt)

        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        all_val_preds[va_idx] += predict(model, va_loader,   DEVICE, desc=f"{fold_desc} best val")
        all_test_preds        += predict(model, test_loader, DEVICE, desc=f"{fold_desc} test")

    all_test_preds /= FOLDS
    print(f"\nOOF log-loss: {compute_log_loss(labels_arr, all_val_preds):.4f}")

    sub = pd.DataFrame(all_test_preds, columns=LABEL_COLS)
    sub.insert(0, "id", test_df["id"].values)
    sub.to_csv(OUTPUT, index=False)
    print(f"\nSubmission saved to {OUTPUT}")
    print(sub.head())
    assert (sub[LABEL_COLS].sum(axis=1) - 1.0).abs().max() < 1e-4, "Probabilities don't sum to 1!"
    print("Probability sum check passed.")


if __name__ == "__main__":
    main()
