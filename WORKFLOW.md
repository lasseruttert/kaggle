# Kaggle Code Competition — Local Development Workflow

## Directory Structure

Each competition lives in its own folder. Inside, each model approach is self-contained:

```
<competition>/
  approaches/
    <approach>/
      train.py              # local training script
      inference.py          # Kaggle inference script (pushed as kernel)
      save.py               # load checkpoint(s) → save to kaggle_dataset/
      kernel-metadata.json  # Kaggle kernel config
      run.ps1               # full pipeline: train → save → upload → push → poll → submit
  kaggle_dataset/
    <approach-model>/
      dataset-metadata.json # required by kaggle CLI
      <model files>         # written by save.py
  checkpoints/              # .pt files written by train.py
  <competition-data>/       # downloaded competition data (train.csv, test.csv etc.)
```

---

## One-Time Setup Per Approach

**1. Create the dataset dir and `dataset-metadata.json`:**
```json
{
  "title": "<dataset-title>",
  "id": "<kaggle-username>/<dataset-slug>",
  "licenses": [{"name": "CC0-1.0"}]
}
```

**2. Create `kernel-metadata.json`:**
```json
{
  "id": "<kaggle-username>/<kernel-slug>",
  "title": "<Kernel Title>",
  "code_file": "inference.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": false,
  "competition_sources": ["<competition-slug>"],
  "dataset_sources": ["<kaggle-username>/<dataset-slug>"],
  "kernel_sources": []
}
```

No init step needed for kernels — `kaggle kernels push` creates it on first run.
Dataset is auto-created on first `run.ps1` if it doesn't exist yet (404 fallback to `create`).

---

## run.ps1 Template

```powershell
param(
    [switch]$SkipTrain,
    [string]$Message = "auto deploy <approach> $(Get-Date -Format 'yyyy-MM-dd HH:mm')"
)

$ErrorActionPreference = "Stop"
$root       = Resolve-Path "$PSScriptRoot/../.."
$datasetDir = "$root/kaggle_dataset/<model-dir>"
$kaggleId   = "<kaggle-username>/<kernel-slug>"

if (-not $SkipTrain) {
    Write-Host "=== Training ===" -ForegroundColor Cyan
    conda run --no-capture-output -n <env> python -u "$PSScriptRoot/train.py"
}

Write-Host "=== Saving model ===" -ForegroundColor Cyan
conda run --no-capture-output -n <env> python -u "$PSScriptRoot/save.py"

Write-Host "=== Uploading dataset ===" -ForegroundColor Cyan
# Use Push-Location + "." to avoid conda run splitting on spaces in the path
Push-Location $datasetDir
conda run -n <env> kaggle datasets version -p . -m "$Message" 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Dataset not found, creating..." -ForegroundColor Yellow
    conda run -n <env> kaggle datasets create -p .
    if ($LASTEXITCODE -ne 0) {
        Pop-Location
        Write-Error "Dataset create failed"
    }
}
Pop-Location

Write-Host "=== Pushing kernel ===" -ForegroundColor Cyan
Push-Location $PSScriptRoot
conda run -n <env> kaggle kernels push -p .
Pop-Location

Write-Host "=== Polling kernel status ===" -ForegroundColor Cyan
do {
    Start-Sleep 30
    $status = conda run -n <env> kaggle kernels status $kaggleId
    Write-Host $status
} while ($status -notmatch "complete|error|cancel")

if ($status -match "complete") {
    Write-Host "=== Done! ===" -ForegroundColor Green
    Write-Host "  Code competition: submission is auto-scored on kernel complete."
    Write-Host "  Check: https://www.kaggle.com/competitions/<competition-slug>/submissions"
    Write-Host ""
    Write-Host "  To download output locally:"
    Write-Host "    Push-Location '$PSScriptRoot'; conda run -n <env> kaggle kernels output $kaggleId -p ./output --force; Pop-Location"
} else {
    Write-Host "=== Kernel ended: $status ===" -ForegroundColor Red
    Write-Host "  To fetch output for debugging:"
    Write-Host "    Push-Location '$PSScriptRoot'; conda run -n <env> kaggle kernels output $kaggleId -p ./output --force; Pop-Location"
}
```

**Flags:** `-SkipTrain` skips training (model already trained), `-Message` sets version/submit message.

> **Code vs file-upload competitions:** For code competitions, the kernel run IS the submission — auto-scored when complete. `kaggle competitions submit -f` only works for file-upload competitions.

---

## train.py Conventions

- Use `__file__`-based paths — never relative paths, since `conda run` doesn't cd to the script:
```python
_HERE    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE, "..", "..", "<competition-data>")
CKPT_DIR = os.path.join(_HERE, "..", "..", "checkpoints")
```

- Pre-tokenize the full dataset before training to avoid tokenization bottleneck per batch:
```python
def pretokenize(df, tokenizer, has_labels=True):
    records = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Tokenizing", leave=False):
        enc = tokenizer(...)
        item = {k: torch.tensor(v) for k, v in enc.items()}
        if has_labels:
            item["labels"] = torch.tensor(...)
        records.append(item)
    return records
```

- Use `DataCollatorWithPadding` + `padding=False` in tokenizer calls for dynamic padding per batch (faster than always padding to `MAX_LEN`).

- Mixed precision for speed:
```python
torch.set_float32_matmul_precision("high")  # TF32 on Ampere+

scaler = torch.amp.GradScaler()
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    loss = model(**batch).loss
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

- Use `attn_implementation="sdpa"` on Windows (Triton/torch.compile not available):
```python
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL, num_labels=3, attn_implementation="sdpa", dtype=torch.float32
)
```

- Suppress noisy HuggingFace warnings:
```python
import transformers
transformers.logging.set_verbosity_error()
```

---

## inference.py Conventions

- Auto-detect Kaggle environment and switch all paths:
```python
KAGGLE    = os.path.exists("/kaggle")
MODEL_DIR = "/kaggle/input/<dataset-slug>" if KAGGLE else "G:/path/to/kaggle_dataset/<model-dir>"
DATA_DIR  = "/kaggle/input/competitions/<competition-slug>" if KAGGLE else "G:/path/to/<competition-data>"
OUTPUT    = "/kaggle/working/submission.csv" if KAGGLE else "submission.csv"
```

Note on Kaggle input paths (confirmed empirically):
- **Dataset sources** → `/kaggle/input/<dataset-slug>/` (no username prefix)
- **Competition sources** → `/kaggle/input/competitions/<competition-slug>/` (with `competitions/` prefix)

- Install any runtime deps at the top (Kaggle has no internet after kernel launch):
```python
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "sentencepiece", "-q"], check=True)
```
(Not needed for BERT/RoBERTa — only for DeBERTa/SentencePiece-based tokenizers.)

---

## save.py Conventions

**Simple models** (single checkpoint → `save_pretrained`):
```python
model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu"))
model.save_pretrained(DATASET_DIR)
tokenizer.save_pretrained(DATASET_DIR)
```

**Fold ensembles** (copy `.pt` files directly):
```python
for src in glob.glob(os.path.join(CKPT_DIR, "best_*_s*_f*.pt")):
    shutil.copy2(src, DATASET_DIR)
```

**Averaged folds** (average state dicts → `save_pretrained`):
```python
avg_state = {}
for ckpt in ckpts:
    for k, v in torch.load(ckpt, map_location="cpu").items():
        avg_state[k] = avg_state.get(k, 0) + v.float()
for k in avg_state:
    avg_state[k] /= len(ckpts)
model.load_state_dict(avg_state)
model.save_pretrained(DATASET_DIR)
```

---

## Kaggle CLI Quick Reference

```powershell
# Download competition data
kaggle competitions download -c <competition-slug> -p <dest>

# Dataset management
kaggle datasets create -p <dir>           # first time
kaggle datasets version -p <dir> -m "msg" # subsequent

# Kernel management
kaggle kernels push -p <dir>              # push/update kernel
kaggle kernels status <username>/<slug>   # check status
kaggle kernels output <username>/<slug>   # download output files

# Submit
kaggle competitions submit -c <slug> -f submission.csv -m "msg"
```

---

## Checklist for a New Competition

- [ ] Download competition data: `kaggle competitions download -c <slug>`
- [ ] Create `approaches/<name>/` with `train.py`, `inference.py`, `save.py`, `kernel-metadata.json`, `run.ps1`
- [ ] Create `kaggle_dataset/<model-dir>/dataset-metadata.json`
- [ ] Create `checkpoints/` dir
- [ ] Train: `.\run.ps1` (or `.\run.ps1 -SkipTrain` if checkpoint already exists)
- [ ] Verify submission on Kaggle leaderboard
