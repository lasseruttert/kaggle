"""
save.py — Ensemble fold checkpoints into a single HuggingFace model and save to kaggle_dataset.
Uses the best fold checkpoint (fold 0) or averages state dicts across folds.
Run from approaches/reward/ or anywhere; paths are relative to this file.
"""
import os
import glob
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

HERE        = os.path.dirname(os.path.abspath(__file__))
ROOT        = os.path.join(HERE, "..", "..")
CKPT_DIR    = os.path.join(ROOT, "checkpoints")
DATASET_DIR = os.path.join(ROOT, "kaggle_dataset", "reward-finetuned")
BASE_MODEL  = "OpenAssistant/reward-model-deberta-v3-large-v2"

os.makedirs(DATASET_DIR, exist_ok=True)

pattern = os.path.join(CKPT_DIR, "best_reward_f*.pt")
ckpts   = sorted(glob.glob(pattern))
if not ckpts:
    raise FileNotFoundError(f"No checkpoints matching {pattern}")

print(f"Found {len(ckpts)} fold checkpoints — averaging state dicts ...")

config   = AutoConfig.from_pretrained(BASE_MODEL)
base     = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, dtype=torch.float32)
base.classifier = nn.Linear(config.hidden_size, 3)

# Average state dicts across folds
avg_state = {}
for ckpt in ckpts:
    state = torch.load(ckpt, map_location="cpu")
    for k, v in state.items():
        avg_state[k] = avg_state.get(k, 0) + v.float()
for k in avg_state:
    avg_state[k] /= len(ckpts)

base.load_state_dict(avg_state)
base.save_pretrained(DATASET_DIR)
AutoTokenizer.from_pretrained(BASE_MODEL).save_pretrained(DATASET_DIR)
print(f"Saved averaged model to {DATASET_DIR}")
