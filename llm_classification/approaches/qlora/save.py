"""
save.py — Copy QLoRA checkpoints + tokenizer to kaggle_dataset/qlora-finetuned/.
Run after train.py completes. Run from approaches/qlora/ or anywhere.
"""
import glob
import json
import os
import shutil
from transformers import AutoTokenizer

HERE        = os.path.dirname(os.path.abspath(__file__))
ROOT        = os.path.normpath(os.path.join(HERE, "..", ".."))
CKPT_DIR    = os.path.join(ROOT, "checkpoints")
DATASET_DIR = os.path.join(ROOT, "kaggle_dataset", "qlora-finetuned")

os.makedirs(DATASET_DIR, exist_ok=True)

# Copy LoRA adapter directories (one per fold)
lora_dirs = sorted(glob.glob(os.path.join(CKPT_DIR, "qlora_lora_f*")))
if not lora_dirs:
    raise FileNotFoundError(f"No LoRA adapter dirs found in {CKPT_DIR}. Run train.py first.")
for src in lora_dirs:
    dst = os.path.join(DATASET_DIR, os.path.basename(src))
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    print(f"Copied {src} → {dst}")

# Copy score head checkpoints (one per fold)
score_ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, "best_qlora_score_f*.pt")))
if not score_ckpts:
    raise FileNotFoundError(f"No score checkpoints found in {CKPT_DIR}. Run train.py first.")
for src in score_ckpts:
    dst = os.path.join(DATASET_DIR, os.path.basename(src))
    shutil.copy2(src, dst)
    print(f"Copied {src} → {dst}")

# Save tokenizer so inference.py can load it without internet
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B", use_fast=True)
tokenizer.save_pretrained(DATASET_DIR)
print(f"Tokenizer saved to {DATASET_DIR}")

# Write dataset-metadata.json for kaggle datasets create/version
metadata = {
    "title": "LLM QLoRA Fine-tuned",
    "id": "lasseruttert/llm-qlora-finetuned",
    "licenses": [{"name": "CC0-1.0"}],
}
meta_path = os.path.join(DATASET_DIR, "dataset-metadata.json")
with open(meta_path, "w") as f:
    json.dump(metadata, f, indent=2)
print(f"dataset-metadata.json written to {meta_path}")

print("\nDone. Contents of dataset dir:")
for name in sorted(os.listdir(DATASET_DIR)):
    path = os.path.join(DATASET_DIR, name)
    if os.path.isdir(path):
        n_files = len(os.listdir(path))
        print(f"  {name}/  ({n_files} files)")
    else:
        size_mb = os.path.getsize(path) / 1e6
        print(f"  {name}  ({size_mb:.1f} MB)")
