"""
train.py — Stacked 3-Model Approach
  1. StackedBertModel: RoBERTa-base text-only
  2. StackedFeatModel: 30 hand-crafted features MLP
  3. StackedMetaModel: MLP on [bert_softmax | feat_softmax]
"""

import contextlib
import json
import os
import re
import unicodedata
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, accuracy_score
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
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
BERT_MODEL      = "roberta-base"
MAX_LEN         = 512
FOLDS           = 5
EPOCHS_BERT     = 4
EPOCHS_FEAT     = 30
EPOCHS_META     = 50
LR_BERT         = 2e-5
LR_FEAT         = 1e-3
LR_META         = 1e-3
BATCH_SIZE_BERT = 16
BATCH_SIZE_FEAT = 256
BATCH_SIZE_META = 256
LABEL_SMOOTHING = 0.05
N_FEATURES      = 30

_HERE    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE, "..", "..", "llm-classification-finetuning")
CKPT_DIR = os.path.join(_HERE, "..", "..", "checkpoints")
OUTPUT   = "submission.csv"

torch.set_float32_matmul_precision("high")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(
    f"CUDA available: {torch.cuda.is_available()}"
    + (f"  |  GPU: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "  |  Running on CPU")
)
print(f"Device: {DEVICE}  |  BERT model: {BERT_MODEL}")

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


def tfidf_cosine(doc1: str, doc2: str) -> float:
    """TF-IDF cosine similarity between two documents (pure Python/numpy, no sklearn)."""
    def tokenize(text):
        return re.findall(r"[a-z]{2,}", text.lower())

    words1 = tokenize(doc1)
    words2 = tokenize(doc2)
    if not words1 or not words2:
        return 0.0

    set1, set2 = set(words1), set(words2)
    total1, total2 = len(words1), len(words2)

    freq1: dict = {}
    for w in words1:
        freq1[w] = freq1.get(w, 0) + 1

    freq2: dict = {}
    for w in words2:
        freq2[w] = freq2.get(w, 0) + 1

    vocab = set1 | set2
    vec1, vec2 = [], []
    for w in vocab:
        tf1 = freq1.get(w, 0) / total1
        tf2 = freq2.get(w, 0) / total2
        idf1 = np.log(2 / (1 + float(w in set2))) + 1
        idf2 = np.log(2 / (1 + float(w in set1))) + 1
        vec1.append(tf1 * idf1)
        vec2.append(tf2 * idf2)

    v1 = np.array(vec1, dtype=np.float32)
    v2 = np.array(vec2, dtype=np.float32)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


def build_rich_features(prompt: str, resp_a: str, resp_b: str) -> np.ndarray:
    """Compute 30 hand-crafted features."""
    a, b = resp_a, resp_b

    def word_count(t):     return len(t.split())
    def sent_count(t):     return max(1, len(re.split(r"[.!?]+", t)))
    def md_elements(t):    return len(re.findall(r"^#{1,6}\s|^[-*]\s|^\d+\.", t, re.M))
    def avg_word_len(t):   words = t.split(); return float(np.mean([len(w) for w in words])) if words else 0.0
    def ttr(t):            words = t.lower().split(); return len(set(words)) / max(len(words), 1)
    def bullet_count(t):   return len(re.findall(r"^[-*]\s", t, re.M))
    def numbered_list(t):  return len(re.findall(r"^\d+\.\s", t, re.M))
    def header_count(t):   return len(re.findall(r"^#{1,6}\s", t, re.M))
    def bold_count(t):     return len(re.findall(r"\*\*[^*]+\*\*|__[^_]+__", t))
    def italic_count(t):   return len(re.findall(r"\*[^*]+\*|_[^_]+_", t))
    def para_count(t):     return len([p for p in re.split(r"\n\n+", t.strip()) if p.strip()])

    def fk_grade(t):
        words = t.split()
        if not words:
            return 0.0
        sents = max(1, len(re.split(r"[.!?]+", t)))
        syllables = max(1, sum(len(re.findall(r"[aeiouAEIOU]+", w)) for w in words))
        grade = 0.39 * (len(words) / sents) + 11.8 * (syllables / len(words)) - 15.59
        return float(max(0.0, min(20.0, grade)))

    def avg_sent_len(t):
        sents = [s.strip() for s in re.split(r"[.!?]+", t) if s.strip()]
        return float(np.mean([len(s.split()) for s in sents])) if sents else 0.0

    def sent_len_std(t):
        sents = [s.strip() for s in re.split(r"[.!?]+", t) if s.strip()]
        return float(np.std([len(s.split()) for s in sents])) if len(sents) >= 2 else 0.0

    def sycophancy_opener(t):
        return bool(re.match(
            r"^(great|excellent|sure|of course|absolutely|certainly|good question|"
            r"happy to|i'd be happy|thanks for|thank you)[,!. ]",
            t.strip().lower()
        ))

    def apology_count(t):
        return len(re.findall(
            r"\b(sorry|apologize|apologies|i cannot|i can't|i am unable|unfortunately)\b", t, re.I
        ))

    def num_count(t):   return len(re.findall(r"\b\d+\.?\d*\b", t))
    def url_count(t):   return len(re.findall(r"https?://\S+|www\.\S+", t))
    def hedge_count(t): return len(re.findall(
        r"\b(may|might|could|possibly|perhaps|likely|probably|seem|appear|suggest|indicate)\b", t, re.I
    ))

    la, lb = len(a), len(b)
    total  = la + lb + 1e-9
    wa, wb = word_count(a), word_count(b)

    # Group 1 — Basic length (8)
    f1  = la / total
    f2  = float(la > lb)
    f3  = (wa - wb) / (wa + wb + 1e-9)
    f4  = (sent_count(a) - sent_count(b)) / 10
    f5  = float(a.count("```") > 0) - float(b.count("```") > 0)
    f6  = (md_elements(a) - md_elements(b)) / 5
    f7  = avg_word_len(a) - avg_word_len(b)
    f8  = ttr(a) - ttr(b)

    # Group 2 — Structural markdown (7)
    f9  = (bullet_count(a)  - bullet_count(b))  / 5
    f10 = (numbered_list(a) - numbered_list(b)) / 5
    f11 = (header_count(a)  - header_count(b))  / 3
    f12 = float(re.search(r"\|.+\|", a) is not None) - float(re.search(r"\|.+\|", b) is not None)
    f13 = (bold_count(a)   - bold_count(b))   / 5
    f14 = (italic_count(a) - italic_count(b)) / 5
    f15 = (para_count(a)   - para_count(b))   / 5

    # Group 3 — Readability (3)
    f16 = (fk_grade(a)     - fk_grade(b))     / 5
    f17 = (avg_sent_len(a) - avg_sent_len(b)) / 20
    f18 = (sent_len_std(a) - sent_len_std(b)) / 10

    # Group 4 — LLM anti-patterns (3)
    f19 = float(sycophancy_opener(a))
    f20 = float(sycophancy_opener(b))
    f21 = (apology_count(a) - apology_count(b)) / 3

    # Group 5 — Prompt relevance (3)
    f22 = tfidf_cosine(prompt, a)
    f23 = tfidf_cosine(prompt, b)
    f24 = f22 - f23

    # Group 6 — Cross-response (1)
    words_a = set(re.findall(r"[a-z]{2,}", a.lower()))
    words_b = set(re.findall(r"[a-z]{2,}", b.lower()))
    union_ab = words_a | words_b
    f25 = len(words_a & words_b) / max(len(union_ab), 1)

    # Group 7 — Factual specificity (2)
    f26 = (num_count(a) - num_count(b)) / 5
    f27 = (url_count(a) - url_count(b)) / 3

    # Group 8 — Prompt-aware (2)
    p_words = max(len(prompt.split()), 1)
    f28 = len(a.split()) / p_words / 10
    f29 = len(b.split()) / p_words / 10

    # Group 9 — Hedging (1)
    f30 = (hedge_count(a) - hedge_count(b)) / 5

    return np.array([
        f1,  f2,  f3,  f4,  f5,  f6,  f7,  f8,
        f9,  f10, f11, f12, f13, f14, f15,
        f16, f17, f18,
        f19, f20, f21,
        f22, f23, f24,
        f25,
        f26, f27,
        f28, f29,
        f30,
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
    swapped["label"]          = swapped[LABEL_COLS].values.argmax(axis=1)
    return swapped


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class StackedBertModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            BERT_MODEL, attn_implementation="sdpa", torch_dtype=torch.float32
        )
        hidden_size = self.backbone.config.hidden_size
        self.head = nn.Linear(hidden_size, 3)

    def forward(self, input_ids, attention_mask, labels=None):
        cls = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        logits = self.head(cls)
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)(logits, labels)
        return loss, logits


class StackedFeatModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_FEATURES, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 3),
        )

    def forward(self, features, labels=None):
        logits = self.net(features)
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)(logits, labels)
        return loss, logits


class StackedMetaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )

    def forward(self, x, labels=None):
        logits = self.net(x)
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)(logits, labels)
        return loss, logits


# ---------------------------------------------------------------------------
# Datasets / Collators
# ---------------------------------------------------------------------------

def pretokenize_bert(df: pd.DataFrame, tokenizer, has_labels: bool = True) -> list:
    records = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Tokenizing BERT", leave=False):
        prompt     = clean_text(parse_prompt(row["prompt"]))
        resp_a_raw = clean_text(parse_prompt(row["response_a"]))
        resp_b_raw = clean_text(parse_prompt(row["response_b"]))
        prompt_t, resp_a_t, resp_b_t = truncate_parts(
            tokenizer, prompt, resp_a_raw, resp_b_raw, MAX_LEN
        )
        enc = tokenizer(
            prompt_t,
            f"Response A: {resp_a_t} Response B: {resp_b_t}",
            max_length=MAX_LEN,
            padding=False,
            truncation=True,
        )
        item = {k: torch.tensor(v) for k, v in enc.items() if k != "token_type_ids"}
        if has_labels:
            item["labels"] = torch.tensor(
                int(np.argmax(row[LABEL_COLS].values.astype(float))), dtype=torch.long
            )
        records.append(item)
    return records


class BertDataset(Dataset):
    def __init__(self, records: list):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return dict(self.records[idx])


class BertCollator:
    def __init__(self, tokenizer):
        self._pad = DataCollatorWithPadding(tokenizer)

    def __call__(self, features):
        return self._pad(features)


class FeatureDataset(Dataset):
    def __init__(self, features_np: np.ndarray, labels_np: np.ndarray = None):
        self.features = torch.tensor(features_np, dtype=torch.float32)
        self.labels   = torch.tensor(labels_np, dtype=torch.long) if labels_np is not None else None

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        item = {"features": self.features[idx]}
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item


class MetaDataset(Dataset):
    def __init__(self, m1_preds_np: np.ndarray, m2_preds_np: np.ndarray, labels_np: np.ndarray = None):
        m1 = torch.tensor(m1_preds_np, dtype=torch.float32)
        m2 = torch.tensor(m2_preds_np, dtype=torch.float32)
        self.x      = torch.cat([m1, m2], dim=1)  # [N, 6]
        self.labels = torch.tensor(labels_np, dtype=torch.long) if labels_np is not None else None

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        item = {"x": self.x[idx]}
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item


# ---------------------------------------------------------------------------
# Training helpers — BERT
# ---------------------------------------------------------------------------

def train_epoch_bert(model, loader, optimizer, scheduler, amp_scaler, device, epoch, total_epochs):
    model.train()
    total_loss, n_steps = 0.0, 0
    use_cuda = device.type == "cuda"
    pbar = tqdm(loader, desc=f"BERT Epoch {epoch}/{total_epochs} [train]", leave=False)
    for batch in pbar:
        batch  = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop("labels")
        ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_cuda else contextlib.nullcontext()
        with ctx:
            loss, _ = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=labels,
            )
        if torch.isnan(loss):
            optimizer.zero_grad()
            continue
        amp_scaler.scale(loss).backward()
        amp_scaler.unscale_(optimizer)
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(grad_norm):
            optimizer.zero_grad()
            amp_scaler.update()
            continue
        amp_scaler.step(optimizer)
        amp_scaler.update()
        scheduler.step()
        optimizer.zero_grad()
        total_loss += loss.item()
        n_steps    += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / max(n_steps, 1)


@torch.no_grad()
def predict_bert(model, loader, device, desc="BERT Predicting"):
    model.eval()
    all_logits = []
    use_cuda = device.type == "cuda"
    for batch in tqdm(loader, desc=desc, leave=False):
        batch.pop("labels", None)
        batch = {k: v.to(device) for k, v in batch.items()}
        ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_cuda else contextlib.nullcontext()
        with ctx:
            _, logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
        all_logits.append(logits.cpu().float())
    return torch.cat(all_logits).softmax(-1).numpy()


# ---------------------------------------------------------------------------
# Training helpers — Feat
# ---------------------------------------------------------------------------

def train_epoch_feat(model, loader, optimizer, device, epoch, total_epochs):
    model.train()
    total_loss, n_steps = 0.0, 0
    pbar = tqdm(loader, desc=f"Feat Epoch {epoch}/{total_epochs} [train]", leave=False)
    for batch in pbar:
        batch  = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop("labels")
        optimizer.zero_grad()
        loss, _ = model(batch["features"], labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_steps    += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / max(n_steps, 1)


@torch.no_grad()
def predict_feat(model, loader, device, desc="Feat Predicting"):
    model.eval()
    all_logits = []
    for batch in tqdm(loader, desc=desc, leave=False):
        batch.pop("labels", None)
        batch = {k: v.to(device) for k, v in batch.items()}
        _, logits = model(batch["features"])
        all_logits.append(logits.cpu().float())
    return torch.cat(all_logits).softmax(-1).numpy()


# ---------------------------------------------------------------------------
# Training helpers — Meta
# ---------------------------------------------------------------------------

def train_epoch_meta(model, loader, optimizer, device):
    model.train()
    total_loss, n_steps = 0.0, 0
    for batch in loader:
        batch  = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop("labels")
        optimizer.zero_grad()
        loss, _ = model(batch["x"], labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_steps    += 1
    return total_loss / max(n_steps, 1)


@torch.no_grad()
def predict_meta(model, loader, device, desc="Meta Predicting"):
    model.eval()
    all_logits = []
    for batch in tqdm(loader, desc=desc, leave=False):
        batch.pop("labels", None)
        batch = {k: v.to(device) for k, v in batch.items()}
        _, logits = model(batch["x"])
        all_logits.append(logits.cpu().float())
    return torch.cat(all_logits).softmax(-1).numpy()


# ---------------------------------------------------------------------------
# Feature matrix builder
# ---------------------------------------------------------------------------

def build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    feats = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building features", leave=False):
        prompt = clean_text(parse_prompt(row["prompt"]))
        resp_a = clean_text(parse_prompt(row["response_a"]))
        resp_b = clean_text(parse_prompt(row["response_b"]))
        feats.append(build_rich_features(prompt, resp_a, resp_b))
    return np.stack(feats)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test_df  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

    train_df["label"] = train_df[LABEL_COLS].values.argmax(axis=1)
    aug_df = pd.concat([train_df, swap_ab(train_df)], ignore_index=True)
    print(f"Train: {len(train_df)}  Augmented: {len(aug_df)}  Test: {len(test_df)}")

    # ------------------------------------------------------------------
    # Pre-tokenize test (BERT) and compute feature matrices
    # ------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
    collator  = BertCollator(tokenizer)

    print("Pre-tokenizing test set for BERT...")
    test_bert_records = pretokenize_bert(test_df, tokenizer, has_labels=False)
    test_bert_loader  = DataLoader(
        BertDataset(test_bert_records),
        batch_size=BATCH_SIZE_BERT * 2, shuffle=False, pin_memory=True, collate_fn=collator,
    )

    print("Building feature matrices...")
    test_feat_raw  = build_feature_matrix(test_df)   # [N_test,  30], unscaled
    train_feat_raw = build_feature_matrix(train_df)  # [N_train, 30], unscaled

    # ------------------------------------------------------------------
    # Fold splitters
    # ------------------------------------------------------------------
    bert_kf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)
    feat_kf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

    bert_splits = list(bert_kf.split(aug_df,   aug_df["label"]))
    feat_splits = list(feat_kf.split(train_df, train_df["label"]))

    # ------------------------------------------------------------------
    # OOF / test accumulator arrays
    # ------------------------------------------------------------------
    oof_bert = np.zeros((len(aug_df),   3), dtype=np.float32)
    oof_feat = np.zeros((len(train_df), 3), dtype=np.float32)

    test_bert_preds = np.zeros((len(test_df), 3), dtype=np.float32)
    test_feat_preds = np.zeros((len(test_df), 3), dtype=np.float32)

    os.makedirs(CKPT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # K-Fold loop
    # ------------------------------------------------------------------
    for fold in range(FOLDS):
        print(f"\n{'='*60}")
        print(f"FOLD {fold+1}/{FOLDS}")
        print(f"{'='*60}")

        # ---- BERT ----
        bert_tr_idx, bert_va_idx = bert_splits[fold]
        bert_tr_df = aug_df.iloc[bert_tr_idx].reset_index(drop=True)
        bert_va_df = aug_df.iloc[bert_va_idx].reset_index(drop=True)
        print(f"  BERT — train: {len(bert_tr_df)}  val: {len(bert_va_df)}")

        bert_train_recs = pretokenize_bert(bert_tr_df, tokenizer)
        bert_val_recs   = pretokenize_bert(bert_va_df, tokenizer)

        bert_train_loader = DataLoader(
            BertDataset(bert_train_recs), batch_size=BATCH_SIZE_BERT,
            shuffle=True, pin_memory=True, collate_fn=collator,
        )
        bert_val_loader = DataLoader(
            BertDataset(bert_val_recs), batch_size=BATCH_SIZE_BERT * 2,
            shuffle=False, pin_memory=True, collate_fn=collator,
        )

        model_bert     = StackedBertModel().to(DEVICE)
        amp_scaler     = torch.amp.GradScaler()
        optimizer_bert = AdamW(model_bert.parameters(), lr=LR_BERT, weight_decay=0.01)
        total_steps    = len(bert_train_loader) * EPOCHS_BERT
        scheduler_bert = get_linear_schedule_with_warmup(
            optimizer_bert, int(0.1 * total_steps), total_steps
        )

        bert_va_labels = bert_va_df["label"].values
        bert_ckpt_path = os.path.join(CKPT_DIR, f"best_stacked_bert_f{fold}.pt")
        best_bert_loss = float("inf")

        for epoch in range(1, EPOCHS_BERT + 1):
            train_loss = train_epoch_bert(
                model_bert, bert_train_loader, optimizer_bert, scheduler_bert,
                amp_scaler, DEVICE, epoch, EPOCHS_BERT,
            )
            val_preds = predict_bert(
                model_bert, bert_val_loader, DEVICE,
                desc=f"BERT Epoch {epoch}/{EPOCHS_BERT} [val]",
            )
            val_loss = compute_log_loss(bert_va_labels, val_preds)
            val_acc  = accuracy_score(bert_va_labels, val_preds.argmax(axis=1))
            print(f"  BERT Epoch {epoch}/{EPOCHS_BERT}  train_loss={train_loss:.4f}  "
                  f"val_log_loss={val_loss:.4f}  val_acc={val_acc:.4f}")
            if val_loss < best_bert_loss:
                best_bert_loss = val_loss
                torch.save(model_bert.state_dict(), bert_ckpt_path)
                print(f"    ↳ Saved BERT checkpoint (val_log_loss={best_bert_loss:.4f})")

        model_bert.load_state_dict(torch.load(bert_ckpt_path, map_location=DEVICE))
        oof_bert[bert_va_idx]  = predict_bert(model_bert, bert_val_loader, DEVICE,
                                              desc=f"BERT Fold {fold+1} OOF")
        test_bert_preds       += predict_bert(model_bert, test_bert_loader, DEVICE,
                                              desc=f"BERT Fold {fold+1} test") / FOLDS
        del model_bert
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

        # ---- Feat ----
        feat_tr_idx, feat_va_idx = feat_splits[fold]
        feat_tr_X_raw = train_feat_raw[feat_tr_idx]
        feat_va_X_raw = train_feat_raw[feat_va_idx]
        feat_tr_y     = train_df["label"].values[feat_tr_idx]
        feat_va_y     = train_df["label"].values[feat_va_idx]
        print(f"  Feat — train: {len(feat_tr_idx)}  val: {len(feat_va_idx)}")

        scaler      = StandardScaler()
        feat_tr_X   = scaler.fit_transform(feat_tr_X_raw)
        feat_va_X   = scaler.transform(feat_va_X_raw)
        test_feat_X = scaler.transform(test_feat_raw)

        feat_train_loader = DataLoader(
            FeatureDataset(feat_tr_X, feat_tr_y),
            batch_size=BATCH_SIZE_FEAT, shuffle=True, pin_memory=True,
        )
        feat_val_loader = DataLoader(
            FeatureDataset(feat_va_X, feat_va_y),
            batch_size=BATCH_SIZE_FEAT * 2, shuffle=False, pin_memory=True,
        )
        feat_test_loader = DataLoader(
            FeatureDataset(test_feat_X),
            batch_size=BATCH_SIZE_FEAT * 2, shuffle=False, pin_memory=True,
        )

        model_feat     = StackedFeatModel().to(DEVICE)
        optimizer_feat = AdamW(model_feat.parameters(), lr=LR_FEAT, weight_decay=1e-4)
        feat_ckpt_path = os.path.join(CKPT_DIR, f"best_stacked_feat_f{fold}.pt")
        best_feat_loss = float("inf")

        for epoch in range(1, EPOCHS_FEAT + 1):
            train_loss = train_epoch_feat(
                model_feat, feat_train_loader, optimizer_feat, DEVICE, epoch, EPOCHS_FEAT,
            )
            val_preds = predict_feat(model_feat, feat_val_loader, DEVICE,
                                     desc=f"Feat Epoch {epoch}/{EPOCHS_FEAT} [val]")
            val_loss  = compute_log_loss(feat_va_y, val_preds)
            if epoch % 5 == 0 or epoch == 1:
                val_acc = accuracy_score(feat_va_y, val_preds.argmax(axis=1))
                print(f"  Feat Epoch {epoch}/{EPOCHS_FEAT}  train_loss={train_loss:.4f}  "
                      f"val_log_loss={val_loss:.4f}  val_acc={val_acc:.4f}")
            if val_loss < best_feat_loss:
                best_feat_loss = val_loss
                torch.save({
                    "state_dict":   model_feat.state_dict(),
                    "scaler_mean":  scaler.mean_,
                    "scaler_scale": scaler.scale_,
                }, feat_ckpt_path)

        print(f"  Feat best val_log_loss={best_feat_loss:.4f} → {feat_ckpt_path}")

        ckpt_bundle = torch.load(feat_ckpt_path, map_location=DEVICE)
        model_feat.load_state_dict(ckpt_bundle["state_dict"])
        oof_feat[feat_va_idx]  = predict_feat(model_feat, feat_val_loader, DEVICE,
                                              desc=f"Feat Fold {fold+1} OOF")
        test_feat_preds       += predict_feat(model_feat, feat_test_loader, DEVICE,
                                              desc=f"Feat Fold {fold+1} test") / FOLDS
        del model_feat
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Meta OOF alignment
    # ------------------------------------------------------------------
    # aug_df[:len(train_df)] == train_df (originals are the first half)
    oof_bert_orig = oof_bert[:len(train_df)]
    oof_bert_loss = compute_log_loss(train_df["label"].values, oof_bert_orig)
    oof_feat_loss = compute_log_loss(train_df["label"].values, oof_feat)
    print(f"\nOOF BERT log-loss (orig): {oof_bert_loss:.4f}")
    print(f"OOF Feat log-loss:        {oof_feat_loss:.4f}")

    # ------------------------------------------------------------------
    # Meta training
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Training Meta-Learner")
    print(f"{'='*60}")

    meta_labels  = train_df["label"].values
    full_meta_ds = MetaDataset(oof_bert_orig, oof_feat, meta_labels)
    n_meta_val   = max(1, int(0.2 * len(full_meta_ds)))
    n_meta_train = len(full_meta_ds) - n_meta_val
    meta_train_ds, meta_val_ds = random_split(
        full_meta_ds, [n_meta_train, n_meta_val],
        generator=torch.Generator().manual_seed(42),
    )

    meta_train_loader = DataLoader(meta_train_ds, batch_size=BATCH_SIZE_META,
                                   shuffle=True, pin_memory=True)
    meta_val_loader   = DataLoader(meta_val_ds,   batch_size=BATCH_SIZE_META,
                                   shuffle=False, pin_memory=True)
    meta_test_loader  = DataLoader(
        MetaDataset(test_bert_preds, test_feat_preds),
        batch_size=BATCH_SIZE_META, shuffle=False,
    )

    model_meta     = StackedMetaModel().to(DEVICE)
    optimizer_meta = AdamW(model_meta.parameters(), lr=LR_META, weight_decay=1e-4)
    meta_ckpt_path = os.path.join(CKPT_DIR, "best_stacked_meta.pt")
    best_meta_loss = float("inf")

    for epoch in range(1, EPOCHS_META + 1):
        train_loss = train_epoch_meta(model_meta, meta_train_loader, optimizer_meta, DEVICE)

        # Inline val loop (also collect labels for log-loss)
        model_meta.eval()
        val_logits_list, val_labels_list = [], []
        with torch.no_grad():
            for batch in meta_val_loader:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                lbl   = batch.pop("labels")
                _, logits = model_meta(batch["x"])
                val_logits_list.append(logits.cpu().float())
                val_labels_list.append(lbl.cpu())
        val_preds_meta  = torch.cat(val_logits_list).softmax(-1).numpy()
        val_labels_meta = torch.cat(val_labels_list).numpy()
        val_loss        = compute_log_loss(val_labels_meta, val_preds_meta)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Meta Epoch {epoch}/{EPOCHS_META}  train_loss={train_loss:.4f}  "
                  f"val_log_loss={val_loss:.4f}")
        if val_loss < best_meta_loss:
            best_meta_loss = val_loss
            torch.save(model_meta.state_dict(), meta_ckpt_path)

    print(f"Meta best val_log_loss={best_meta_loss:.4f} → {meta_ckpt_path}")

    # ------------------------------------------------------------------
    # Final predictions
    # ------------------------------------------------------------------
    model_meta.load_state_dict(torch.load(meta_ckpt_path, map_location=DEVICE))
    test_final_preds = predict_meta(model_meta, meta_test_loader, DEVICE, desc="Meta test")

    sub = pd.DataFrame(test_final_preds, columns=LABEL_COLS)
    sub.insert(0, "id", test_df["id"].values)
    sub.to_csv(OUTPUT, index=False)
    print(f"\nSubmission saved to {OUTPUT}")
    print(sub.head())
    assert (sub[LABEL_COLS].sum(axis=1) - 1.0).abs().max() < 1e-4, "Probabilities don't sum to 1!"
    print("Probability sum check passed.")


if __name__ == "__main__":
    main()
