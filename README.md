# DeepU-PRS

Nonlinear and uncertainty-aware SNP-heritability modeling for annotation-informed polygenic risk scoring.

DeepU-PRS uses a deep ensemble of feed-forward networks over per-SNP functional annotations to predict a heritability prior **h²_j** and an aleatoric variance **σ²_j** for every SNP. The variance is then folded into a hyperparameter-free closed-form shrinkage — **DeepU-PRS (Adaptive)** — that is passed to LDAK MegaPRS (BayesR) for posterior effect-size estimation.

```
GWAS summary stats + functional annotations
             │
             ▼
   ┌──────────────────────────┐
   │  DeepU-PRS network  (×M) │   train.py                (one seed at a time)
   │  in : a_j (65 or 205 dim)│                           M-ensemble
   │  out: (h²_j, log σ²_j)   │
   └──────────────────────────┘
             │
             ▼
   ┌──────────────────────────┐
   │ Ensemble aggregation     │   make_ivw_prior.py
   │   IVW h²                 │   (precision-weighted mean of h²
   │   simple-mean σ²         │    + arithmetic mean of σ²)
   └──────────────────────────┘
             │
             ▼
   ┌──────────────────────────┐
   │ Adaptive shrinkage       │   make_adaptive_prior.py
   │ h²_adapt = h²_IVW ·      │
   │   σ²_med / (σ²+σ²_med)   │
   └──────────────────────────┘
             │
             ▼
   ┌──────────────────────────┐
   │ LDAK --mega-prs bayesr   │   external (Speed et al.)
   └──────────────────────────┘
             │
             ▼
        posterior β̂ (effects)
```

## Adaptive shrinkage

The principal deliverable of DeepU-PRS is a per-SNP heritability prior that has already absorbed the ensemble aleatoric variance through a closed-form Bayesian shrinkage:

> **h²_adapt,j = h²_IVW,j · ( σ²_med / (σ²_j + σ²_med) )**

- **h²_IVW,j** — inverse-variance-weighted ensemble mean of per-SNP heritability across the M seeds.
- **σ²_j** — simple mean of the ensemble aleatoric variance (`σ² = exp(log σ²)`) across seeds.
- **σ²_med** — trait-specific median of σ²_j across all SNPs.

Properties:
- Monotone in σ²_j — noisier SNPs shrink harder.
- Scale-equivariant — only the ratio σ²_j / σ²_med enters, so the shrinkage is invariant to global rescaling.
- Hyperparameter-free — no validation tuning, no test leakage.
- Edge behavior: σ²_j ≪ σ²_med → factor ≈ 1 (keep), σ²_j = σ²_med → factor = 0.5, σ²_j ≫ σ²_med → factor ≈ 0 (shrink to zero).

## Repository layout

```
DeepU-PRS/
├── README.md                    # this file
├── requirements.txt             # Python deps
├── train.py                     # Train one ensemble member
├── make_ivw_prior.py            # M-seed aggregation: IVW h² + simple-mean σ²
├── make_adaptive_prior.py       # Closed-form adaptive shrinkage
└── scripts/
    └── run_ensemble.sh          # End-to-end wrapper for one trait
```

## Dependencies

See `requirements.txt`. Tested on Python 3.11, PyTorch 2.0+

## Input files

Every path below is passed to `train.py` via CLI.

| Flag | Format | Description |
|------|--------|-------------|
| `--data_root` | directory | Root that holds `<file_path>/neale.train.summaries` and per-chr tensor caches (auto-created on the first run) |
| `--file_path` | subdir name | Trait label used as a subdir under `data_root` (e.g. `HDL`) |
| `--biomarker` | string | Lowercase alias printed in logs (e.g. `hdl`) |
| `--pca` | bool | `True` → 205-feature input (functional + Enformer PCA); `False` → 65-feature |
| `--annot_205` | csv | Per-SNP 205-feature annotation matrix. Same shape convention. Used when `--pca True`. |
| `--maf` | plink `.frq` | Reference allele-frequency file with columns `CHR SNP A1 A2 MAF NCHROBS` |
| `--ld_root` | directory | Contains `<r2_coverage>/chrld_<c>.npy` (pair,pair,R²) and `<r2_coverage>/chr<c>_edge_index.npy` (2 × N_pairs) for each chromosome |
| `--r2_coverage` | subdir name | LD subdir label (default `cut_0.01`) |
| `--lr` | float | Learning rate (paper uses `0.001`) |
| `--seed` | int | Ensemble-member seed |
| `--ver` | string | Version tag embedded in output filenames (paper uses `18607`) |

## LD reference files (edge index and R²)
**`chr{N}_edge_index.npy`** — 2D array of SNP index pairs for LD-linked SNPs on chromosome N
- Shape: `(N_pairs, 2)`, dtype `int64`
- Column 0: index of SNP_A. **The index is the 0-indexed position of that SNP within the per-chromosome subset of the GWAS summary statistics file**
- Column 1: index of SNP_B (same convention).
- Only SNP pairs with R² above the coverage threshold (default `cut_0.01`, i.e., R² > 0.01) are included.
- Because the indices reference the summary-statistics SNP order, the annotation matrix and MAF file must be aligned to that same per-chromosome SNP order at training time. `train.py` performs this alignment internally by filtering both to the summary SNP set.

**`chrld_{N}.npy`** — accompanying R² values as SNP-ID triples
- Shape: `(N_pairs, 3)`, dtype `object`
- Column 0: SNP_A predictor (string, format `<chr>:<pos>`)
- Column 1: SNP_B predictor (string, same format)
- Column 2: R² value (float, > 0.01)

The GWAS summary statistics file itself is expected at `<data_root>/<file_path>/neale.train.summaries` in LDAK whitespace format (`Predictor A1 A2 Direction Stat n`).

Because the input files are large, they are hosted separately from the code

## Data

Input annotations and example priors: **[10.5281/zenodo.21341238](https://doi.org/10.5281/zenodo.21341238)** 

- `annot_all_mmscaled.csv.parquet` — per-SNP annotation matrix (`Predictor` + 205 features). Pass via `--annot_205`.
- `HDL_r2-cut_0.01_lr0.001_noamb.ind.her.pos.adaptive` (+ `_snps.adaptive.txt`) — example DeepU-PRS (Adaptive) SNP heritability prior for HDL, ready for LDAK MegaPRS.

## Quick start (HDL example)

```bash
# 1. Train M=5 ensemble members, one per seed.
for s in seed_1~seed_N; do
  python -u train.py \
      --data_root  ./data \
      --file_path  HDL \
      --biomarker  hdl \
      --pca        True \
      --annot_205  ./data/annot_205.csv \
      --maf        ./data/plink.frq \
      --ld_root    ./ld \
      --r2_coverage cut_0.01 \
      --lr         0.001 \
      --seed       ${s} \
      --ver        test1
done
# per-SNP h² and σ² are written under ./data/HDL/annot205/

# 2. Aggregate across the ensemble: IVW h² + simple-mean σ².
python make_ivw_prior.py \
    --input_dir   ./data/HDL/annot205 \
    --seeds       seed_1~seed_N \
    --ver         test1 \
    --lr          0.001 \
    --r2_coverage cut_0.01 \
    --out_dir     ./data/HDL/annot205

# 3. Adaptive shrinkage → LDAK-ready heritability prior.
python make_adaptive_prior.py \
    --h2_ivw  ./data/HDL/annot205/h2_ivw.ind.her.pos.ens_ivw \
    --sigma2  ./data/HDL/annot205/sigma2_simple.ind.var.ens_simple \
    --out_dir ./data/HDL/annot205

# 4. LDAK MegaPRS BayesR with the adaptive prior.
ldak --mega-prs ./data/HDL/annot205/bayesr_adaptive \
     --model bayesr \
     --ind-hers ./data/HDL/annot205/_noamb.ind.her.pos.adaptive \
     --summary  ./data/HDL/neale.train.summaries \
     --cors     ./ld/HDL_cors \
     --high-LD  ./ld/highld.snplist \
     --extract  ./data/HDL/annot205/_noamb_snps.adaptive.txt \
     --allow-ambiguous NO --window-cm 1 --cv-proportion .1

# 5. Score on a held-out cohort.
ldak --calc-scores ./data/HDL/annot205/scores_adaptive \
     --bfile     ./data/holdout_cohort \
     --scorefile ./data/HDL/annot205/bayesr_adaptive.effects \
     --power 0 \
     --summary   ./data/HDL/neale.test.summaries \
     --allow-ambiguous NO
```

`scripts/run_ensemble.sh` bundles steps 1–3 into a single call.

## Citation

*DeepU-PRS: Nonlinear and Uncertainty-Aware SNP Heritability Modeling for Annotation-Informed Polygenic Risk Scoring*.
