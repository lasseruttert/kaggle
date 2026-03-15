"""
save.py — Copy stacked checkpoints + tokenizer/config to kaggle_dataset.
Run from approaches/stacked/ or anywhere; paths are relative to this file.
"""
import glob
import os
import shutil
from transformers import AutoTokenizer, AutoConfig

HERE        = os.path.dirname(os.path.abspath(__file__))
ROOT        = os.path.join(HERE, "..", "..")
CKPT_DIR    = os.path.join(ROOT, "checkpoints")
DATASET_DIR = os.path.join(ROOT, "kaggle_dataset", "stacked-finetuned")
BASE_MODEL  = "roberta-base"

os.makedirs(DATASET_DIR, exist_ok=True)

# Copy BERT fold checkpoints
bert_ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, "best_stacked_bert_f*.pt")))
if not bert_ckpts:
    raise FileNotFoundError(f"No BERT fold checkpoints found in {CKPT_DIR}. Run train.py first.")
for src in bert_ckpts:
    dst = os.path.join(DATASET_DIR, os.path.basename(src))
    shutil.copy2(src, dst)
    print(f"Copied {src} → {dst}")

# Copy Feat fold checkpoints (include scaler bundles)
feat_ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, "best_stacked_feat_f*.pt")))
if not feat_ckpts:
    raise FileNotFoundError(f"No Feat fold checkpoints found in {CKPT_DIR}. Run train.py first.")
for src in feat_ckpts:
    dst = os.path.join(DATASET_DIR, os.path.basename(src))
    shutil.copy2(src, dst)
    print(f"Copied {src} → {dst}")

# Copy meta checkpoint
meta_ckpt = os.path.join(CKPT_DIR, "best_stacked_meta.pt")
if not os.path.exists(meta_ckpt):
    raise FileNotFoundError(f"Meta checkpoint not found: {meta_ckpt}. Run train.py first.")
shutil.copy2(meta_ckpt, os.path.join(DATASET_DIR, "best_stacked_meta.pt"))
print(f"Copied {meta_ckpt} → {DATASET_DIR}/best_stacked_meta.pt")

# Save tokenizer and config so inference.py can reconstruct the BERT model
AutoTokenizer.from_pretrained(BASE_MODEL).save_pretrained(DATASET_DIR)
AutoConfig.from_pretrained(BASE_MODEL).save_pretrained(DATASET_DIR)
print(f"Tokenizer + config saved to {DATASET_DIR}")
print("Done.")
