# DeepU-PRS

**Uncertainty-aware Deep Prior Estimation for Polygenic Risk Scoring (PRS)**

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-orange)](https://pytorch.org/)

## Overview

**DeepU-PRS** is a deep learning framework designed to improve Polygenic Risk Scoring by incorporating nonlinear functional annotations and modeling aleatoric uncertainty. This repository provides a cleaned, reproducible training script and minimal scaffolding to run DeepU‑PRS.

> **Key Idea:** A fully connected (FC) network predicts SNP-level prior means and log-variances (aleatoric uncertainty) from functional annotations. These priors are propagated through sparse Linkage Disequilibrium (LD) structures to obtain architecture-aware effect-size priors for downstream Bayesian PRS models.

### Key Features
* **Aleatoric Uncertainty Modeling:** Predicts mean and log-variance with **deep ensemble support** (run the script multiple times with different `--seed`).
* **Sparse LD Integration:** Efficiently applies chromosome-wise sparse LD matrices.
* **Architecture-Aware:** Outputs per-chromosome priors and a combined CSV for downstream analysis.

---

## System Requirements

* **Operating System:** Linux (Recommended) or macOS
* **Hardware:** NVIDIA GPU recommended for training (tested on RTX A5000, 24GB VRAM)
* **Software:**
    * Python 3.8+
    * PyTorch (CUDA support recommended)
    * NumPy, Pandas, Scikit-learn
    * [LDAK](https://dougspeed.com/ldak/) (for downstream PRS calculation)

---

## Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Junnh/DeepU-PRS.git](https://github.com/Junnh/DeepU-PRS.git)
    cd DeepU-PRS
    ```

2.  **Set up the environment:**
    We recommend using Conda.
    ```bash
    conda create -n deepu-prs python=3.9
    conda activate deepu-prs
    pip install torch numpy pandas scipy
    ```

---

## 1) Repository Layout & Data Expectations

DeepU-PRS expects a specific directory layout for annotations, summary statistics, and LD files.

```text
<base_dir>/
├── annotations/
│   └── annotation.csv               # Per-SNP features (Predictor, feat1..featK)
├── summaries/
│   ├── train.summaries              # Summary stats for training (tab-delimited)
│   └── test.summaries               # Summary stats for evaluation (tab-delimited)
├── maf/
│   └── plink.frq                    # Reference MAF (CHR, SNP, MAF...)
└── ld/
```

### Required Columns
Ensure your input files contain the following columns:

* **`annotations/annotation.csv`**
    * Required: `Predictor`, `feat1`, `feat2`, ... (and other annotation features)
    * *Note:* If the `chr` column is missing, the script will automatically extract it by parsing the `Predictor` column (expected format: `chr:pos:ref:alt`).
* **`summaries/*.summaries`**
    * Required: `Predictor` (SNP ID), `Stat` (Test statistic), `n` (Sample size).
* **`maf/plink.frq`**
    * Required: `CHR`, `SNP`, `MAF`.

> **Important:** The LD files are *looked up only*, not created by this repository. Ensure your precomputed LD files use SNP identifiers consistent with those in your `summaries` file.

---

## 2) Training Script

The main training script is `deepu_prs_refactored.py`.

```bash
python deepu_prs_refactored.py \
  --file_path HDL \
  --pca true \
  --lr 1e-3 \
  --seed 2025 \
  --ver v1 \
  --base_dir /path/to/base_dir \
  --output_dir /path/to/output_dir
```
### Arguments

* `--file_path` *(str, required)*: used only to name the output subfolder.
* `--pca` *(bool)*: if `true`, use the deeper FC head (“enformer” mode); else compact head (“deep\_imp0”).
* `--lr` *(float, required)*: learning rate for **LAMB** (built‑in minimal implementation).
* `--seed` *(int, required)*: random seed.
* `--ver` *(str)*: version tag for output filenames. Default `v1`.
* `--base_dir` *(path)*: directory containing the data layout shown above.
* `--output_dir` *(path)*: where to save artifacts. Defaults to `<base_dir>/<file_path>`.
* (advanced) `--epochs`, `--patience`, `--warmup_epochs`, `--weight_decay`, `--clip_grad_norm`, `--min_var`, `--max_var`.

### Outputs

Files are written under:

```
<output_dir>/<mode>/
  where <mode> = enformer (if --pca true) else deep_imp0
```

with filename prefix:

```
f{VER}_lr{LR}_seed{SEED}
```

Exported files:

* `*.noamb.ind.her` (TSV, no header): `Predictor\tHeritability`
* `*.noamb.ind.her.pos` (TSV, no header): only positive heritabilities
* `*.noamb_snps.txt` (one column): SNPs used in the positive set
* `*.noamb.ind.her.logvar.csv` (CSV): `Predictor,Heritability,logvar`

---

## 3) LDAK evaluation: end‑to‑end shell example

Below is a **generic** workflow for training effects via LDAK `--mega-prs` (BayesR) and computing scores on a held‑out summary set. Replace paths in brackets with your own.

> **Assumptions**
>
> * LDAK binary at `$LDAK_BIN` (e.g., `/path/to/ldak` or `/path/to/ldak.out`).
> * Correlation and high‑LD resources are prepared (`--cors`, `--high-LD`).
> * Training/test summary files at `<base_dir>/summaries/{train,test}.summaries`.

```bash
#!/usr/bin/env bash
set -euo pipefail

# ---- user config ----
BASE_DIR="/path/to/base_dir"          # data root (see layout)
OUT_ROOT="/path/to/output_root"       # where deepu_prs_refactored.py writes outputs
LDAK_BIN="/path/to/ldak.out"          # ldak executable
BFILE="/path/to/plink_prefix"         # plink bed/bim/fam without extension
CORS_DIR="/path/to/cors"              # ldak --cors directory
HIGHLD_FILE="/path/to/highld.predictors"  # file passed to --high-LD / --exclude

TRAIT="HDL"            # used only for naming subfolder
PCA_MODE="enformer"    # "enformer" (pca=true) or "deep_imp0" (pca=false)
LR=0.001
VER="v1"
SEEDS=(2023)

# optional: run the training script (one seed shown)
for SEED in "${SEEDS[@]}"; do
  python deepu_prs_refactored.py \
    --file_path "$TRAIT" \
    --pca $([[ "$PCA_MODE" == "enformer" ]] && echo true || echo false) \
    --lr "$LR" \
    --seed "$SEED" \
    --ver "$VER" \
    --base_dir "$BASE_DIR" \
    --output_dir "$OUT_ROOT"

done

# LDAK: fit BayesR and compute PRS scores per seed
for SEED in "${SEEDS[@]}"; do
  MODE_DIR="$OUT_ROOT/$([[ "$PCA_MODE" == "enformer" ]] && echo enformer || echo deep_imp0)"
  PREFIX="${MODE_DIR}/f${VER}_lr${LR}_seed${SEED}"

  # 1) Fit effects with BayesR using per‑SNP priors
  "$LDAK_BIN" --mega-prs "${MODE_DIR}/f${VER}_bayesr_lr${LR}_seed${SEED}" \
    --model bayesr \
    --ind-hers   "${PREFIX}.noamb.ind.her.pos" \
    --summary    "${BASE_DIR}/summaries/train.summaries" \
    --cors       "$CORS_DIR" \
    --cv-proportion .1 \
    --high-LD    "$HIGHLD_FILE" \
    --window-cm  1 \
    --extract    "${PREFIX}.noamb_snps.txt" \
    --allow-ambiguous NO

  # 2) Score on a held‑out set (effects produced by the step above)
  "$LDAK_BIN" --calc-scores "${MODE_DIR}/f${VER}_scores_lr${LR}_seed${SEED}" \
    --bfile     "$BFILE" \
    --scorefile "${MODE_DIR}/f${VER}_bayesr_lr${LR}_seed${SEED}.effects" \
    --power     0 \
    --summary   "${BASE_DIR}/summaries/test.summaries" \
    --allow-ambiguous NO \
    --exclude   "$HIGHLD_FILE"

done
```

### Notes & mapping to outputs

* `--ind-hers` and `--extract` point to the files emitted by this repo.
* You control seed sweeps with `SEEDS=(...)`.
* `PCA_MODE` controls which subfolder the training writes to, and later which subfolder LDAK reads from.
* `--summary` uses your own training/test sets; the script only reads `train.summaries`.

---

## 5) Tips & troubleshooting

* **Shapes**: Ensure `edge_index.npy` has shape `(2, E)` (else it’s transposed). `chrld_*.npy` should have three columns: `PredictorA, PredictorB, r2`.
* **SNP ID consistency**: The intersection logic keeps LD edges only if **both SNPs** exist in `summaries` for that chromosome.
* **Guard on `Stat`**: Non‑positive `Stat` is clamped to `1e-6`.
* **Output filtration**: `*.noamb.ind.her.pos` keeps only positive prior means, commonly used for LDAK BayesR.

---

## 6) Citation & license
