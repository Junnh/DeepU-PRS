# DeepU-PRS

Uncertainty-aware deep prior estimation for polygenic risk scoring (PRS).
This repository provides a cleaned, reproducible training script and minimal scaffolding to run DeepU‑PRS.

> Key idea: a fully connected (FC) network predicts SNP-level prior means and log-variances (aleatoric uncertainty) from functional annotations. Priors are propagated through sparse LD to obtain architecture-aware effect-size priors for PRS.

---

## Features
- Aleatoric-uncertainty head (mean + logvar) with **deep ensemble support** (run the script multiple times with different `--seed`).
- Chromosome-wise **sparse LD application**.
- **Warmup**, **ReduceLROnPlateau** scheduler, **early stopping**, gradient clipping.
- Deterministic seeds, structured logging, and argumentized paths (no hard-coded local paths).
- Outputs per-chromosome priors and a combined CSV.

---

## Quick start (real data)

```
python deepu_prs_train.py   --file_path "HDL"   --biomarker "HDL"   --lr 1e-3   --r2_coverage "cut_0.01/cross_chr"   --pca   --seed 2025   --ver fc205_a5000   --path_ref_maf ./ref/plink.frq   --path_annot_pca ./annot/annot_all_mmscaled.csv   --path_annot_default ./annot/annot_imp_v2.csv   --path_summaries_train "./summaries/{file_path}/neale.train.summaries"   --path_ld_edge_attr "./ld/{r2_coverage}/chrld_{chrom}.npy"   --path_edge_index "./edge/{r2_coverage}/chr{chrom}_edge_index.npy"   --out_dir ./outputs/hdl_fc205
```

### Expected inputs
- **Annotation CSV** (either 65 or 205 features). Must contain `Predictor` column with `chr:pos:ref:alt` (or similar) and a `chr` column added internally.
- **Summary statistics** (`neale.train.summaries`-like; whitespace-delimited) with columns including `Predictor`, `Stat`, and `n`.
- **Ref MAF** (`plink.frq`) with `CHR`, `SNP`, `MAF` columns.
- **Sparse LD** for each chromosome:
  - Edge weights: `./ld/<r2_coverage>/chrld_<chrom>.npy` (Nx3 with i, j, r)
  - Edge index: `./edge/<r2_coverage>/chr<chrom>_edge_index.npy` (shape (2, E) or (E, 2))

> Outputs are saved in `--out_dir`: `model_best.pt`, `*_prior_chr{N}.pt`, `*_prior_logvar_chr{N}.pt`, and a combined `*_ind_her_and_logvar.csv`.

---

## Citation

---


