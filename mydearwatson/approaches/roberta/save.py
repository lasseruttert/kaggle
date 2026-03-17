"""
save.py — Copy RoBERTa seed checkpoints + tokenizer/config to kaggle_dataset.
"""
import glob
import json
import os
import shutil
from transformers import AutoTokenizer, AutoConfig

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
CKPT_DIR = os.path.join(ROOT, "checkpoints")
DATASET_DIR = os.path.join(ROOT, "kaggle_dataset", "watson-roberta-finetuned")
BASE_MODEL = "roberta-base"

os.makedirs(DATASET_DIR, exist_ok=True)

# Copy seed checkpoints
ckpt_files = sorted(glob.glob(os.path.join(CKPT_DIR, "seed_*_best.pt")))
if not ckpt_files:
    raise FileNotFoundError(f"No seed checkpoints found in {CKPT_DIR}. Run train.py first.")

for src in ckpt_files:
    dst = os.path.join(DATASET_DIR, os.path.basename(src))
    shutil.copy2(src, dst)
    print(f"Copied {src} -> {dst}")

# Save tokenizer and config so inference.py can reconstruct the model
AutoTokenizer.from_pretrained(BASE_MODEL).save_pretrained(DATASET_DIR)
AutoConfig.from_pretrained(BASE_MODEL, num_labels=3).save_pretrained(DATASET_DIR)
print(f"Tokenizer + config saved to {DATASET_DIR}")

# Write dataset-metadata.json
metadata = {
    "title": "Watson RoBERTa Fine-tuned",
    "id": "lasseruttert/watson-roberta-finetuned",
    "licenses": [{"name": "CC0-1.0"}],
}
meta_path = os.path.join(DATASET_DIR, "dataset-metadata.json")
with open(meta_path, "w") as f:
    json.dump(metadata, f, indent=2)
print(f"dataset-metadata.json written to {meta_path}")

print("Done.")
