"""
inference.py — RoBERTa NLI inference for Contradictory, My Dear Watson.
Loads seed checkpoints and produces submission.csv.
Designed to run on Kaggle with no internet access.

Expected Kaggle dataset layout:
  /kaggle/input/watson-roberta-finetuned/
      seed_*_best.pt
      config.json
      tokenizer_config.json
      ...
"""

import glob
import os
import numpy as np
import pandas as pd
import torch
import transformers
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from tqdm import tqdm

transformers.logging.set_verbosity_error()

# --- Config ---
KAGGLE = os.path.exists("/kaggle")
MODEL_DIR = "/kaggle/input/datasets/lasseruttert/watson-roberta-finetuned" if KAGGLE else "G:/My Drive/kaggle/mydearwatson/kaggle_dataset/watson-roberta-finetuned"
DATA_DIR = "/kaggle/input/competitions/contradictory-my-dear-watson" if KAGGLE else "G:/My Drive/kaggle/mydearwatson/contradictory-my-dear-watson"
OUTPUT = "/kaggle/working/submission.csv" if KAGGLE else "submission.csv"
MAX_LEN = 128
BATCH_SIZE = 32
CKPT_PATTERN = os.path.join(MODEL_DIR, "seed_*_best.pt")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"CUDA available: {torch.cuda.is_available()}" + (f"  |  GPU: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "  |  Running on CPU"))
print(f"Device: {DEVICE}  |  Model dir: {MODEL_DIR}")


def main():
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    print(f"Test rows: {len(test_df)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    config = AutoConfig.from_pretrained(MODEL_DIR)

    test_premises = test_df["premise"].fillna("").tolist()
    test_hypotheses = test_df["hypothesis"].fillna("").tolist()
    test_encodings = tokenizer(
        test_premises, test_hypotheses,
        truncation=True, padding="max_length", max_length=MAX_LEN,
        return_tensors="pt",
    )

    ckpt_paths = sorted(glob.glob(CKPT_PATTERN))
    if not ckpt_paths:
        raise FileNotFoundError(f"No seed checkpoints found at {CKPT_PATTERN}")
    print(f"Found {len(ckpt_paths)} seed checkpoint(s): {[os.path.basename(p) for p in ckpt_paths]}")

    all_logits = []
    for ckpt_path in ckpt_paths:
        print(f"Loading {os.path.basename(ckpt_path)} ...")
        model = AutoModelForSequenceClassification.from_config(config)
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
        model.to(DEVICE)
        model.eval()

        seed_logits = []
        with torch.no_grad():
            for i in tqdm(range(0, len(test_premises), BATCH_SIZE), desc="Inference"):
                input_ids = test_encodings["input_ids"][i:i+BATCH_SIZE].to(DEVICE)
                attention_mask = test_encodings["attention_mask"][i:i+BATCH_SIZE].to(DEVICE)
                with torch.autocast(device_type="cuda", dtype=torch.float16) if DEVICE.type == "cuda" else __import__("contextlib").nullcontext():
                    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                seed_logits.append(logits.cpu())

        all_logits.append(torch.cat(seed_logits, dim=0))
        del model
        torch.cuda.empty_cache()

    avg_logits = torch.stack(all_logits, dim=0).mean(dim=0)
    preds = avg_logits.argmax(dim=1).tolist()

    submission = pd.DataFrame({"id": test_df["id"], "prediction": preds})
    submission.to_csv(OUTPUT, index=False)
    print(f"\nSubmission saved to {OUTPUT}")
    print(submission["prediction"].value_counts().sort_index())


if __name__ == "__main__":
    main()
