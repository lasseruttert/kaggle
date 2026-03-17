import re
import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

# --- Preprocessing ---
PREPROCESS = True

def preprocess(text):
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[!]{2,}", "!", text)
    text = re.sub(r"[?]{2,}", "?", text)
    text = re.sub(r"[.]{3,}", "...", text)
    text = re.sub(r"#(\w+)", lambda m: m.group(1).replace("_", " "), text)
    text = re.sub(r"can't\b", "cannot", text)
    text = re.sub(r"n't\b", " not", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# --- Config ---
MODEL_NAME = "roberta-base"
MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 3
LR = 2e-5
SEED = 42
N_FOLDS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Dataset ---
class TweetDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(texts, truncation=True, padding="max_length", max_length=MAX_LEN, return_tensors="pt")
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}, self.labels[idx]

# --- Load & prepare data ---
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

raw_texts = train_df["text"].fillna("").tolist()
keywords = train_df["keyword"].fillna("").tolist()
if PREPROCESS:
    raw_texts = [preprocess(t) for t in raw_texts]
texts = [f"{kw} {t}".strip() if kw else t for kw, t in zip(keywords, raw_texts)]
labels = train_df["target"].tolist()

raw_test_texts = test_df["text"].fillna("").tolist()
test_keywords = test_df["keyword"].fillna("").tolist()
if PREPROCESS:
    raw_test_texts = [preprocess(t) for t in raw_test_texts]
test_texts = [f"{kw} {t}".strip() if kw else t for kw, t in zip(test_keywords, raw_test_texts)]

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
test_encodings = tokenizer(test_texts, truncation=True, padding="max_length", max_length=MAX_LEN, return_tensors="pt")

texts_arr = np.array(texts)
labels_arr = np.array(labels)

# --- 5-Fold weighted ensemble ---
os.makedirs("checkpoints", exist_ok=True)
fold_results = []  # list of (val_f1, test_logits)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

for fold, (train_idx, val_idx) in enumerate(skf.split(texts_arr, labels_arr)):
    print(f"\n{'='*40}\nFold {fold+1}/{N_FOLDS}\n{'='*40}")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    train_texts = texts_arr[train_idx].tolist()
    val_texts = texts_arr[val_idx].tolist()
    train_labels = labels_arr[train_idx].tolist()
    val_labels = labels_arr[val_idx].tolist()

    train_dataset = TweetDataset(train_texts, train_labels, tokenizer)
    val_dataset = TweetDataset(val_texts, val_labels, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2, torch_dtype=torch.float32)
    model.to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=LR)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps)

    best_val_f1 = 0.0
    ckpt_path = f"checkpoints/fold_{fold+1}_best.pt"
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch, labels_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels_batch = labels_batch.clone().detach().long().to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels_batch)
            loss = outputs.loss
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

        model.eval()
        all_preds, all_labels_val = [], []
        with torch.no_grad():
            for batch, labels_batch in val_loader:
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels_batch = labels_batch.clone().detach().long().to(DEVICE)
                preds = model(input_ids=input_ids, attention_mask=attention_mask).logits.argmax(dim=1)
                all_preds.extend(preds.cpu().tolist())
                all_labels_val.extend(labels_batch.cpu().tolist())

        val_f1 = f1_score(all_labels_val, all_preds)
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f} | Val F1: {val_f1:.4f}")
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), ckpt_path)

    # Collect test logits for this fold
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()
    fold_logits = []
    with torch.no_grad():
        for i in range(0, len(test_texts), BATCH_SIZE):
            input_ids = test_encodings["input_ids"][i:i+BATCH_SIZE].to(DEVICE)
            attention_mask = test_encodings["attention_mask"][i:i+BATCH_SIZE].to(DEVICE)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            fold_logits.append(logits.cpu())
    fold_results.append((best_val_f1, torch.cat(fold_logits, dim=0)))

# --- Weighted ensemble ---
print("\n--- Fold Results ---")
for i, (f1, _) in enumerate(fold_results):
    print(f"  Fold {i+1}: Val F1 = {f1:.4f}")

f1_weights = [f1 for f1, _ in fold_results]
total_weight = sum(f1_weights)
print(f"\nWeights (normalized): {[f'{w/total_weight:.4f}' for w in f1_weights]}")

weighted_logits = sum(w * logits for (w, logits) in zip(f1_weights, [r[1] for r in fold_results]))
avg_logits = weighted_logits / total_weight
preds = avg_logits.argmax(dim=1).tolist()

submission = pd.DataFrame({"id": test_df["id"], "target": preds})
submission.to_csv("submission.csv", index=False)
print("Saved submission.csv")
