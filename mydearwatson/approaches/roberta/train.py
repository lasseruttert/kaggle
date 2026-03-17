import os
import random
import numpy as np
import pandas as pd
import torch
import transformers
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from tqdm import tqdm

transformers.logging.set_verbosity_error()

# --- Config ---
MODEL_NAME = "roberta-base"
MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 3
LR = 2e-5
SEEDS = [42, 43, 44]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
DATA_DIR = os.path.join(ROOT, "contradictory-my-dear-watson")
CKPT_DIR = os.path.join(ROOT, "checkpoints")


# --- Dataset ---
class NLIDataset(Dataset):
    def __init__(self, premises, hypotheses, labels, tokenizer):
        self.encodings = tokenizer(
            premises, hypotheses,
            truncation=True, padding="max_length", max_length=MAX_LEN,
            return_tensors="pt",
        )
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}, self.labels[idx]


# --- Load data ---
train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

premises = train_df["premise"].fillna("").tolist()
hypotheses = train_df["hypothesis"].fillna("").tolist()
labels = train_df["label"].tolist()

test_premises = test_df["premise"].fillna("").tolist()
test_hypotheses = test_df["hypothesis"].fillna("").tolist()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
test_encodings = tokenizer(
    test_premises, test_hypotheses,
    truncation=True, padding="max_length", max_length=MAX_LEN,
    return_tensors="pt",
)

# --- Ensemble loop ---
os.makedirs(CKPT_DIR, exist_ok=True)
all_test_logits = []
scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")

for seed in SEEDS:
    print(f"\n{'='*40}\nSeed {seed}\n{'='*40}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    train_premises, val_premises, train_hyps, val_hyps, train_labels, val_labels = train_test_split(
        premises, hypotheses, labels, test_size=0.1, random_state=seed, stratify=labels,
    )

    train_dataset = NLIDataset(train_premises, train_hyps, train_labels, tokenizer)
    val_dataset = NLIDataset(val_premises, val_hyps, val_labels, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=3, torch_dtype=torch.float32,
        attn_implementation="sdpa",
    )
    model.to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=LR)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps)

    best_val_acc = 0.0
    ckpt_path = os.path.join(CKPT_DIR, f"seed_{seed}_best.pt")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch, labels_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels_batch = labels_batch.clone().detach().long().to(DEVICE)

            with torch.autocast(device_type=DEVICE.type, dtype=torch.float16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels_batch)
                loss = outputs.loss

            total_loss += loss.item()
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch, labels_batch in val_loader:
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels_batch = labels_batch.clone().detach().long().to(DEVICE)
                preds = model(input_ids=input_ids, attention_mask=attention_mask).logits.argmax(dim=1)
                correct += (preds == labels_batch).sum().item()
                total += labels_batch.size(0)

        val_acc = correct / total
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), ckpt_path)

    print(f"Best Val Acc for seed {seed}: {best_val_acc:.4f}")

    # Collect test logits
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    model.eval()
    seed_logits = []
    with torch.no_grad():
        for i in range(0, len(test_premises), BATCH_SIZE):
            input_ids = test_encodings["input_ids"][i:i+BATCH_SIZE].to(DEVICE)
            attention_mask = test_encodings["attention_mask"][i:i+BATCH_SIZE].to(DEVICE)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            seed_logits.append(logits.cpu())
    all_test_logits.append(torch.cat(seed_logits, dim=0))

# --- Ensemble & Submission ---
avg_logits = torch.stack(all_test_logits, dim=0).mean(dim=0)
preds = avg_logits.argmax(dim=1).tolist()

submission = pd.DataFrame({"id": test_df["id"], "prediction": preds})
submission.to_csv(os.path.join(ROOT, "submission.csv"), index=False)
print(f"\nSaved submission.csv ({len(submission)} rows)")
print(submission["prediction"].value_counts().sort_index())
