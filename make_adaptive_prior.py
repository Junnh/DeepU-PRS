"""Adaptive aleatoric-shrinkage prior — the final-deliverable DeepU-PRS prior.

Closed-form shrinkage of the IVW heritability prior by the per-SNP aleatoric
variance estimated by the DeepU ensemble:

      h2_adaptive_j = h2_IVW_j  *  ( sigma2_median / (sigma2_j + sigma2_median) )

Properties:
  - monotone in sigma2_j      (more noise -> less prior weight)
  - scale-equivariant         (only the *ratio* sigma2_j / sigma2_median matters)
  - fixed point sigma2_j = sigma2_median  =>  factor = 0.5
  - zero tunable hyperparameters  =>  no cohort-specific tuning, no test leakage

Edge behavior:
  - sigma2_j  <<  sigma2_median  ->  factor ~ 1     (kept ~as-is)
  - sigma2_j  ==  sigma2_median  ->  factor = 0.5   (halved)
  - sigma2_j  >>  sigma2_median  ->  factor ~ 0     (shrunk toward zero)

Inputs (LDAK whitespace-tabular, no header):
  --h2_ivw    file with columns:  Predictor  h2
              -> IVW heritability prior from make_ivw_prior.py (positive subset only)
  --sigma2    file with columns:  Predictor  sigma2
              -> simple-mean ensemble aleatoric variance, per SNP

Outputs (also LDAK format, no header):
  {out_dir}/_noamb.ind.her.adaptive        ALL SNPs (Predictor, h2_adaptive)
  {out_dir}/_noamb.ind.her.pos.adaptive    positive subset
  {out_dir}/_noamb_snps.adaptive.txt       predictor list of the positive subset

Usage:
  python make_adaptive_prior.py \
      --h2_ivw  path/to/h2_ivw.ind.her.pos.ens_ivw \
      --sigma2  path/to/sigma2_simple.ind.var.ens_simple \
      --out_dir path/to/out/
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--h2_ivw',  required=True, help='IVW h^2 prior file (Predictor, h2)')
    p.add_argument('--sigma2',  required=True, help='aleatoric sigma^2 file (Predictor, sigma2)')
    p.add_argument('--out_dir', required=True, help='output directory')
    p.add_argument('--prefix',  default='',     help='optional output filename prefix '
                                                    '(default: empty -> _noamb.ind.her.adaptive)')
    args = p.parse_args()

    if not os.path.isfile(args.h2_ivw):
        sys.exit(f"missing IVW prior: {args.h2_ivw}")
    if not os.path.isfile(args.sigma2):
        sys.exit(f"missing sigma2 file: {args.sigma2}")
    os.makedirs(args.out_dir, exist_ok=True)

    print("=== Adaptive shrinkage prior ===")
    ivw = pd.read_csv(args.h2_ivw, sep=r"\s+", header=None, names=["Predictor", "h2"], engine="c")
    sig = pd.read_csv(args.sigma2, sep=r"\s+", header=None, names=["Predictor", "sigma2"], engine="c")
    print(f"  IVW positive rows: {len(ivw):,}   sigma rows: {len(sig):,}")

    merged = ivw.merge(sig, on="Predictor", how="left")
    n_miss = int(merged["sigma2"].isna().sum())
    if n_miss:
        med0 = float(np.nanmedian(merged["sigma2"].values))
        merged["sigma2"] = merged["sigma2"].fillna(med0)
        print(f"  filled {n_miss:,} missing sigma2 with median {med0:.4e}")

    sig_med = float(np.median(merged["sigma2"].values))
    sig_p5  = float(np.percentile(merged['sigma2'],  5))
    sig_p95 = float(np.percentile(merged['sigma2'], 95))
    print(f"  sigma2_median (baseline): {sig_med:.4e}")
    print(f"  sigma2  p5={sig_p5:.4e}  med={sig_med:.4e}  p95={sig_p95:.4e}")

    # === Adaptive shrinkage (THE formula) ===
    factor = sig_med / (merged["sigma2"].values + sig_med)
    merged["h2_adaptive"] = merged["h2"].values * factor

    # Diagnostics
    f_p5  = float(np.percentile(factor,  5))
    f_med = float(np.median(factor))
    f_p95 = float(np.percentile(factor, 95))
    print(f"  shrink factor  p5={f_p5:.3f}  med={f_med:.3f}  p95={f_p95:.3f}  "
          f"min={factor.min():.3f}  max={factor.max():.3f}")
    h2_in  = merged["h2"].values
    h2_out = merged["h2_adaptive"].values
    ratio  = float(h2_out.sum() / h2_in.sum()) if h2_in.sum() > 0 else float('nan')
    print(f"  h2_IVW    sum={h2_in.sum():.4f}  mean={h2_in.mean():.3e}")
    print(f"  h2_adapt  sum={h2_out.sum():.4f}  mean={h2_out.mean():.3e}  (sum-ratio: {ratio:.3f})")

    # Write LDAK .ind.her format (no header, tab-separated)
    pref = (args.prefix + "_") if args.prefix else ""
    out_all  = os.path.join(args.out_dir, f"{pref}_noamb.ind.her.adaptive")
    out_pos  = os.path.join(args.out_dir, f"{pref}_noamb.ind.her.pos.adaptive")
    out_snps = os.path.join(args.out_dir, f"{pref}_noamb_snps.adaptive.txt")

    df_all = pd.DataFrame({"Predictor": merged["Predictor"].values, "Heritability": h2_out})
    df_all.to_csv(out_all, sep="\t", index=False, header=False)
    pos = df_all[df_all["Heritability"] > 0]
    pos.to_csv(out_pos, sep="\t", index=False, header=False)
    pos[["Predictor"]].to_csv(out_snps, sep="\t", index=False, header=False)

    print(f"\n  -> {out_all}   ({len(df_all):,} SNPs)")
    print(f"  -> {out_pos}    ({len(pos):,} SNPs)")
    print(f"  -> {out_snps}   ({len(pos):,} SNPs)")
    print("\nFeed the .ind.her.pos.adaptive file to LDAK MegaPRS via --ind-hers.")


if __name__ == "__main__":
    main()
