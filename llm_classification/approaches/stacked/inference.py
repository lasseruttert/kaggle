"""
inference.py — Stacked 3-Model Inference
Load fine-tuned BERT + Feat fold checkpoints, run meta-learner, produce submission.
Designed to run on Kaggle with no internet access.

Expected dataset layout (/kaggle/input/llm-stacked-finetuned/):
    best_stacked_bert_f0.pt ... best_stacked_bert_f4.pt
    best_stacked_feat_f0.pt ... best_stacked_feat_f4.pt   (bundles with scaler)
    best_stacked_meta.pt
    config.json
    tokenizer_config.json
    ...
"""

import contextlib
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
from transformers import AutoTokenizer, AutoModel, AutoConfig, DataCollatorWithPadding
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
KAGGLE    = os.path.exists("/kaggle")
MODEL_DIR = (
    "/kaggle/input/llm-stacked-finetuned"
    if KAGGLE else
    "G:/My Drive/kaggle/llm_classification/kaggle_dataset/stacked-finetuned"
)
DATA_DIR  = (
    "/kaggle/input/competitions/llm-classification-finetuning"
    if KAGGLE else
    "G:/My Drive/kaggle/llm_classification/llm-classification-finetuning"
)
OUTPUT    = "/kaggle/working/submission.csv" if KAGGLE else "submission.csv"

MAX_LEN      = 512
BATCH_SIZE   = 16
N_FEATURES   = 30
LABEL_SMOOTHING = 0.05

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(
    f"CUDA available: {torch.cuda.is_available()}"
    + (f"  |  GPU: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "  |  Running on CPU")
)
print(f"Device: {DEVICE}  |  Model dir: {MODEL_DIR}")

LABEL_COLS = ["winner_model_a", "winner_model_b", "winner_tie"]


# ---------------------------------------------------------------------------
# Helpers  (verbatim from train.py)
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


def truncate_parts(tokenizer, prompt: str, resp_a: str, resp_b: str, max_len: int):
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

    f1  = la / total
    f2  = float(la > lb)
    f3  = (wa - wb) / (wa + wb + 1e-9)
    f4  = (sent_count(a) - sent_count(b)) / 10
    f5  = float(a.count("```") > 0) - float(b.count("```") > 0)
    f6  = (md_elements(a) - md_elements(b)) / 5
    f7  = avg_word_len(a) - avg_word_len(b)
    f8  = ttr(a) - ttr(b)
    f9  = (bullet_count(a)  - bullet_count(b))  / 5
    f10 = (numbered_list(a) - numbered_list(b)) / 5
    f11 = (header_count(a)  - header_count(b))  / 3
    f12 = float(re.search(r"\|.+\|", a) is not None) - float(re.search(r"\|.+\|", b) is not None)
    f13 = (bold_count(a)   - bold_count(b))   / 5
    f14 = (italic_count(a) - italic_count(b)) / 5
    f15 = (para_count(a)   - para_count(b))   / 5
    f16 = (fk_grade(a)     - fk_grade(b))     / 5
    f17 = (avg_sent_len(a) - avg_sent_len(b)) / 20
    f18 = (sent_len_std(a) - sent_len_std(b)) / 10
    f19 = float(sycophancy_opener(a))
    f20 = float(sycophancy_opener(b))
    f21 = (apology_count(a) - apology_count(b)) / 3
    f22 = tfidf_cosine(prompt, a)
    f23 = tfidf_cosine(prompt, b)
    f24 = f22 - f23
    words_a  = set(re.findall(r"[a-z]{2,}", a.lower()))
    words_b  = set(re.findall(r"[a-z]{2,}", b.lower()))
    union_ab = words_a | words_b
    f25 = len(words_a & words_b) / max(len(union_ab), 1)
    f26 = (num_count(a) - num_count(b)) / 5
    f27 = (url_count(a) - url_count(b)) / 3
    p_words = max(len(prompt.split()), 1)
    f28 = len(a.split()) / p_words / 10
    f29 = len(b.split()) / p_words / 10
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
# Models  (verbatim from train.py, except StackedBertModel takes model_dir)
# ---------------------------------------------------------------------------

class StackedBertModel(nn.Module):
    def __init__(self, model_dir: str):
        super().__init__()
        config = AutoConfig.from_pretrained(model_dir)
        config.attn_implementation = "sdpa"
        self.backbone = AutoModel.from_config(config)
        self.head = nn.Linear(config.hidden_size, 3)

    def forward(self, input_ids, attention_mask):
        cls = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        return self.head(cls)


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

    def forward(self, features):
        return self.net(features)


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

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Datasets  (verbatim from train.py)
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
    def __init__(self, features_np: np.ndarray):
        self.features = torch.tensor(features_np, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {"features": self.features[idx]}


class MetaDataset(Dataset):
    def __init__(self, m1_preds_np: np.ndarray, m2_preds_np: np.ndarray):
        m1 = torch.tensor(m1_preds_np, dtype=torch.float32)
        m2 = torch.tensor(m2_preds_np, dtype=torch.float32)
        self.x = torch.cat([m1, m2], dim=1)  # [N, 6]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return {"x": self.x[idx]}


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
# Inference helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_bert(model, loader, device, desc="BERT Inference"):
    model.eval()
    all_logits = []
    use_cuda = device.type == "cuda"
    for batch in tqdm(loader, desc=desc, leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_cuda else contextlib.nullcontext()
        with ctx:
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        all_logits.append(logits.cpu().float())
    return torch.cat(all_logits).softmax(-1).numpy()


@torch.no_grad()
def predict_feat(model, loader, device, desc="Feat Inference"):
    model.eval()
    all_logits = []
    for batch in tqdm(loader, desc=desc, leave=False):
        batch  = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch["features"])
        all_logits.append(logits.cpu().float())
    return torch.cat(all_logits).softmax(-1).numpy()


@torch.no_grad()
def predict_meta(model, loader, device, desc="Meta Inference"):
    model.eval()
    all_logits = []
    for batch in tqdm(loader, desc=desc, leave=False):
        batch  = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch["x"])
        all_logits.append(logits.cpu().float())
    return torch.cat(all_logits).softmax(-1).numpy()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    print(f"Test rows: {len(test_df)}")

    # ------------------------------------------------------------------
    # Phase 1 — BERT fold ensemble
    # ------------------------------------------------------------------
    print("\n=== Phase 1: BERT ===")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    collator  = BertCollator(tokenizer)

    test_bert_records = pretokenize_bert(test_df, tokenizer, has_labels=False)
    bert_loader = DataLoader(
        BertDataset(test_bert_records),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collator,
    )

    bert_ckpts = sorted(glob.glob(os.path.join(MODEL_DIR, "best_stacked_bert_f*.pt")))
    if not bert_ckpts:
        raise FileNotFoundError(f"No BERT checkpoints at {MODEL_DIR}/best_stacked_bert_f*.pt")
    print(f"Found {len(bert_ckpts)} BERT fold checkpoint(s)")

    all_bert_preds = []
    for ckpt_path in bert_ckpts:
        print(f"  Loading {os.path.basename(ckpt_path)} ...")
        model_bert = StackedBertModel(MODEL_DIR).to(DEVICE)
        model_bert.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        all_bert_preds.append(predict_bert(model_bert, bert_loader, DEVICE))
        del model_bert
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    bert_avg = np.mean(all_bert_preds, axis=0)  # [N_test, 3]

    # ------------------------------------------------------------------
    # Phase 2 — Feat fold ensemble
    # ------------------------------------------------------------------
    print("\n=== Phase 2: Feature MLP ===")
    test_feat_raw = build_feature_matrix(test_df)  # [N_test, 30], unscaled

    feat_ckpts = sorted(glob.glob(os.path.join(MODEL_DIR, "best_stacked_feat_f*.pt")))
    if not feat_ckpts:
        raise FileNotFoundError(f"No Feat checkpoints at {MODEL_DIR}/best_stacked_feat_f*.pt")
    print(f"Found {len(feat_ckpts)} Feat fold checkpoint(s)")

    all_feat_preds = []
    for ckpt_path in feat_ckpts:
        print(f"  Loading {os.path.basename(ckpt_path)} ...")
        bundle      = torch.load(ckpt_path, map_location=DEVICE)
        scaler_mean  = bundle["scaler_mean"]
        scaler_scale = bundle["scaler_scale"]
        test_feat_X  = (test_feat_raw - scaler_mean) / (scaler_scale + 1e-8)

        feat_loader = DataLoader(
            FeatureDataset(test_feat_X),
            batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=0,
        )

        model_feat = StackedFeatModel().to(DEVICE)
        model_feat.load_state_dict(bundle["state_dict"])
        all_feat_preds.append(predict_feat(model_feat, feat_loader, DEVICE))
        del model_feat
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    feat_avg = np.mean(all_feat_preds, axis=0)  # [N_test, 3]

    # ------------------------------------------------------------------
    # Phase 3 — Meta-learner
    # ------------------------------------------------------------------
    print("\n=== Phase 3: Meta-Learner ===")
    meta_loader = DataLoader(
        MetaDataset(bert_avg, feat_avg),
        batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=0,
    )

    meta_ckpt_path = os.path.join(MODEL_DIR, "best_stacked_meta.pt")
    if not os.path.exists(meta_ckpt_path):
        raise FileNotFoundError(f"Meta checkpoint not found: {meta_ckpt_path}")

    model_meta = StackedMetaModel().to(DEVICE)
    model_meta.load_state_dict(torch.load(meta_ckpt_path, map_location=DEVICE))
    final_preds = predict_meta(model_meta, meta_loader, DEVICE)

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    sub = pd.DataFrame(final_preds, columns=LABEL_COLS)
    sub.insert(0, "id", test_df["id"].values)
    sub.to_csv(OUTPUT, index=False)
    print(f"\nSubmission saved to {OUTPUT}")
    print(sub.head())
    assert (sub[LABEL_COLS].sum(axis=1) - 1.0).abs().max() < 1e-4, "Probabilities don't sum to 1!"
    print("Probability sum check passed.")


if __name__ == "__main__":
    main()
