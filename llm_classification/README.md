# LLM Classification — Codebase Guide

## Directory Structure

```
llm_classification/
├── approaches/
│   ├── basic/
│   │   ├── train.py              # RoBERTa-base + K-fold + swap aug + hand features
│   │   ├── inference.py          # Kaggle inference script (loads fold checkpoints)
│   │   ├── save.py               # Copy fold .pt files + tokenizer/config to kaggle_dataset
│   │   ├── kernel-metadata.json  # Kaggle kernel config
│   │   └── run.ps1               # Full pipeline: train → save → upload → push → poll → submit
│   └── reward/
│       ├── train.py              # OpenAssistant reward backbone → 3-class head
│       ├── inference.py          # Kaggle inference script
│       ├── save.py               # Average fold state dicts → save_pretrained
│       ├── kernel-metadata.json
│       └── run.ps1
├── checkpoints/                  # .pt files produced during training
├── kaggle_dataset/               # Local copies of Kaggle dataset directories
│   ├── bert-finetuned/           # basic fold checkpoints + tokenizer/config
│   └── reward-finetuned/         # reward model (save_pretrained output)
└── llm-classification-finetuning/   # Competition data (train.csv, test.csv)
```

---

## Full Pipeline (PowerShell)

Each approach is self-contained. From the approach folder:

```powershell
# Full pipeline: train + save + upload + push kernel + poll + submit
.\run.ps1

# Skip training (model already trained)
.\run.ps1 -SkipTrain

# Custom message
.\run.ps1 -SkipTrain -Message "v2 larger batch"
```

### Pipeline steps (`run.ps1`)

1. `python train.py` — fine-tune model, saves checkpoint(s) to `../../checkpoints/`
2. `python save.py` — load checkpoint(s), write model files to `../../kaggle_dataset/<dir>/`
3. `kaggle datasets version` — upload dataset to Kaggle (creates on first run if 404)
4. `kaggle kernels push` — push `inference.py` as a Kaggle kernel
5. Poll `kaggle kernels status` until complete/error/cancel
6. Submission is auto-scored on kernel complete (code competition)

---

## Training

### Basic (RoBERTa-base)

```powershell
cd approaches/basic
conda run --no-capture-output -n kaggle python -u train.py
```

Produces `checkpoints/best_basic_f{0..4}.pt` (one per fold).

Config knobs at the top of `train.py`: `FOLDS`, `EPOCHS`, `LR`, `BATCH_SIZE`, `MAX_LEN`.

### Reward model (OpenAssistant backbone)

```powershell
cd approaches/reward
conda run --no-capture-output -n kaggle python -u train.py
```

Produces `checkpoints/best_reward_f{fold}.pt`. Set `INFERENCE_ONLY = True` for zero-shot scoring without fine-tuning.

---

## Saving Models

```powershell
conda run --no-capture-output -n kaggle python -u approaches/basic/save.py   # fold .pt files → kaggle_dataset/bert-finetuned/
conda run --no-capture-output -n kaggle python -u approaches/reward/save.py  # averaged folds → kaggle_dataset/reward-finetuned/
```

---

## Kaggle Dataset & Kernel IDs

| Approach | Dataset | Kernel |
|----------|---------|--------|
| basic    | `lasseruttert/llm-bert-basic-finetuned` | `lasseruttert/bert-basic-inference` |
| reward   | `lasseruttert/llm-reward-finetuned` | `lasseruttert/reward-inference` |

---

## Inference on Kaggle

All `inference.py` scripts auto-detect Kaggle via `os.path.exists("/kaggle")` and switch paths.
Output is written to `/kaggle/working/submission.csv` on Kaggle.

### Kaggle paths used

| Variable | Kaggle path |
|----------|-------------|
| `MODEL_DIR` (basic) | `/kaggle/input/llm-bert-basic-finetuned` |
| `MODEL_DIR` (reward) | `/kaggle/input/llm-reward-finetuned` |
| `DATA_DIR` | `/kaggle/input/competitions/llm-classification-finetuning` |

---

## Notes

- RoBERTa does not use `token_type_ids` — these are dropped from batches before the forward pass.
- `basic/inference.py` averages softmax probabilities across all fold checkpoints found in `MODEL_DIR`.
- `reward/save.py` averages state dicts across all fold checkpoints before saving.
