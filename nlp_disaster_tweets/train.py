import re
import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

# --- Preprocessing ---
PREPROCESS = True

def preprocess(text):
    text = re.sub(r"http\S+|www\.\S+", "", text)   # remove URLs
    text = re.sub(r"@\w+", "", text)                # remove @mentions
    text = re.sub(r"[!]{2,}", "!", text)            # repeated ! -> single
    text = re.sub(r"[?]{2,}", "?", text)            # repeated ? -> single
    text = re.sub(r"[.]{3,}", "...", text)          # 4+ dots -> ellipsis
    text = re.sub(r"#(\w+)", lambda m: m.group(1).replace("_", " "), text)  # #hashtag -> hashtag, #some_tag -> some tag
    text = re.sub(r"can't\b", "cannot", text)        # can't -> cannot
    text = re.sub(r"n't\b", " not", text)            # don't -> do not
    text = re.sub(r"\s+", " ", text).strip()        # collapse whitespace
    # text = text.lower()
    return text

# --- Config ---
MODEL_NAME = "roberta-base"
MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 2
LR = 3e-5
SEEDS = [42, 43, 44]  # one full train run per seed
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

# --- Load & prepare data (done once) ---
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

# --- Ensemble loop ---
os.makedirs("checkpoints", exist_ok=True)
all_test_logits = []

for seed in SEEDS:
    print(f"\n{'='*40}\nSeed {seed}\n{'='*40}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.1, random_state=seed
    )

    train_dataset = TweetDataset(train_texts, train_labels, tokenizer)
    val_dataset = TweetDataset(val_texts, val_labels, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2, torch_dtype=torch.float32)
    model.to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=LR)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps)

    # Train
    best_val_f1 = 0.0
    ckpt_path = f"checkpoints/seed_{seed}_best.pt"
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

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch, labels_batch in val_loader:
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels_batch = labels_batch.clone().detach().long().to(DEVICE)
                preds = model(input_ids=input_ids, attention_mask=attention_mask).logits.argmax(dim=1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels_batch.cpu().tolist())

        val_f1 = f1_score(all_labels, all_preds)
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f} | Val F1: {val_f1:.4f}")
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), ckpt_path)

    # Collect test logits for this seed
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()
    seed_logits = []
    with torch.no_grad():
        for i in range(0, len(test_texts), BATCH_SIZE):
            input_ids = test_encodings["input_ids"][i:i+BATCH_SIZE].to(DEVICE)
            attention_mask = test_encodings["attention_mask"][i:i+BATCH_SIZE].to(DEVICE)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            seed_logits.append(logits.cpu())
    all_test_logits.append(torch.cat(seed_logits, dim=0))

# --- Ensemble & Submission ---
avg_logits = torch.stack(all_test_logits, dim=0).mean(dim=0)
preds = avg_logits.argmax(dim=1).tolist()

submission = pd.DataFrame({"id": test_df["id"], "target": preds})
submission.to_csv("submission.csv", index=False)
print("Saved submission.csv")
