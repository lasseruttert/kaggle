"""
save.py — Copy LLM-CLS checkpoints + tokenizer + config to kaggle_dataset/llm-cls-finetuned/.
Run after train.py completes. Run from approaches/llm_cls/ or anywhere.
"""
import glob
import json
import os
import shutil
from transformers import AutoTokenizer, AutoConfig

HERE        = os.path.dirname(os.path.abspath(__file__))
ROOT        = os.path.normpath(os.path.join(HERE, "..", ".."))
CKPT_DIR    = os.path.join(ROOT, "checkpoints")
DATASET_DIR = os.path.join(ROOT, "kaggle_dataset", "llm-cls-finetuned")
BASE_MODEL  = "Qwen/Qwen2.5-1.5B"

os.makedirs(DATASET_DIR, exist_ok=True)

# Copy state dict checkpoints (one per fold)
ckpt_files = sorted(glob.glob(os.path.join(CKPT_DIR, "best_llm_cls_f*.pt")))
if not ckpt_files:
    raise FileNotFoundError(f"No checkpoints found in {CKPT_DIR}. Run train.py first.")
for src in ckpt_files:
    dst = os.path.join(DATASET_DIR, os.path.basename(src))
    shutil.copy2(src, dst)
    size_mb = os.path.getsize(dst) / 1e6
    print(f"Copied {src} → {dst}  ({size_mb:.0f} MB)")

# Save tokenizer so inference.py can load it without internet
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
tokenizer.save_pretrained(DATASET_DIR)
print(f"Tokenizer saved to {DATASET_DIR}")

# Save config.json so inference.py can rebuild the model architecture without hub access
config = AutoConfig.from_pretrained(BASE_MODEL)
config.save_pretrained(DATASET_DIR)
print(f"config.json saved to {DATASET_DIR}")

# Write dataset-metadata.json for kaggle datasets create/version
metadata = {
    "title": "LLM LLM-CLS Fine-tuned",
    "id": "lasseruttert/llm-llm-cls-finetuned",
    "licenses": [{"name": "CC0-1.0"}],
}
meta_path = os.path.join(DATASET_DIR, "dataset-metadata.json")
with open(meta_path, "w") as f:
    json.dump(metadata, f, indent=2)
print(f"dataset-metadata.json written to {meta_path}")

print("\nDone. Contents of dataset dir:")
for name in sorted(os.listdir(DATASET_DIR)):
    path = os.path.join(DATASET_DIR, name)
    size_mb = os.path.getsize(path) / 1e6
    print(f"  {name}  ({size_mb:.1f} MB)")
