"""
save.py — Copy fold checkpoints + tokenizer/config to kaggle_dataset.
Run from approaches/basic/ or anywhere; paths are relative to this file.
"""
import glob
import os
import shutil
from transformers import AutoTokenizer, AutoConfig

HERE        = os.path.dirname(os.path.abspath(__file__))
ROOT        = os.path.join(HERE, "..", "..")
CKPT_DIR    = os.path.join(ROOT, "checkpoints")
DATASET_DIR = os.path.join(ROOT, "kaggle_dataset", "bert-finetuned")
BASE_MODEL  = "roberta-base"

os.makedirs(DATASET_DIR, exist_ok=True)

# Copy fold checkpoints
ckpt_files = sorted(glob.glob(os.path.join(CKPT_DIR, "best_basic_f*.pt")))
if not ckpt_files:
    raise FileNotFoundError(f"No fold checkpoints found in {CKPT_DIR}. Run train.py first.")

for src in ckpt_files:
    dst = os.path.join(DATASET_DIR, os.path.basename(src))
    shutil.copy2(src, dst)
    print(f"Copied {src} → {dst}")

# Save tokenizer and config so inference.py can reconstruct the model
AutoTokenizer.from_pretrained(BASE_MODEL).save_pretrained(DATASET_DIR)
AutoConfig.from_pretrained(BASE_MODEL).save_pretrained(DATASET_DIR)
print(f"Tokenizer + config saved to {DATASET_DIR}")
print("Done.")
