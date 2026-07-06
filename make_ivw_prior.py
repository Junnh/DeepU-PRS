"""Aggregate M trained DeepU-PRS ensemble members into IVW h^2 prior + simple-mean sigma^2.

Reads the per-seed CSV files produced by train.py:
    <input_dir>/f<VER>_tissue_v37_sub2_r2-<R2CUT>_lr<LR>_noamb.ind.her.logvar<SEED>.csv
Each file has three columns: Predictor, Heritability, logvar

Aggregation across seeds (per SNP j):
    sigma2_{j,m}   = exp(logvar_{j,m})
    h2_IVW,j       = ( sum_m h2_{j,m} / sigma2_{j,m} ) / ( sum_m 1 / sigma2_{j,m} )
    sigma2_simple  = (1/M) * sum_m sigma2_{j,m}

Outputs (LDAK whitespace-tabular, no header):
    <out_dir>/h2_ivw.ind.her.pos.ens_ivw          -- (Predictor, h2_IVW)  positive subset
    <out_dir>/sigma2_simple.ind.var.ens_simple    -- (Predictor, sigma2)  all SNPs

Usage:
    python make_ivw_prior.py \
        --input_dir   ./data/HDL/enformer_new \
        --seeds       2023 2024 2025 2026 2027 \
        --ver         18607 \
        --lr          0.001 \
        --r2_coverage cut_0.01 \
        --out_dir     ./data/HDL/enformer_new
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd


def load_seed(input_dir, ver, r2_coverage, lr, seed):
    fname = (f'f{ver}_tissue_v37_sub2_r2-{r2_coverage}_lr{lr}'
             f'_noamb.ind.her.logvar{seed}.csv')
    path = os.path.join(input_dir, fname)
    if not os.path.isfile(path):
        sys.exit(f'missing per-seed file: {path}')
    df = pd.read_csv(path)
    if not {'Predictor', 'Heritability', 'logvar'}.issubset(df.columns):
        sys.exit(f'expected columns Predictor/Heritability/logvar in {path}, got {list(df.columns)}')
    df = df[['Predictor', 'Heritability', 'logvar']].copy()
    df['sigma2'] = np.exp(df['logvar'].astype(np.float64))
    return df[['Predictor', 'Heritability', 'sigma2']].rename(
        columns={'Heritability': 'h2'})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input_dir',   required=True, help='dir holding train.py per-seed outputs')
    p.add_argument('--seeds',       nargs='+', required=True, help='ensemble seed labels')
    p.add_argument('--ver',         required=True, help='version tag used by train.py')
    p.add_argument('--lr',          required=True, help='learning rate string used by train.py')
    p.add_argument('--r2_coverage', default='cut_0.01', help='LD subdir label used by train.py')
    p.add_argument('--out_dir',     required=True)
    p.add_argument('--eps',         type=float, default=1e-12,
                   help='numerical floor on sigma^2 for IVW weights')
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f'Aggregating M={len(args.seeds)} seeds from {args.input_dir}')
    dfs = [load_seed(args.input_dir, args.ver, args.r2_coverage, args.lr, s) for s in args.seeds]

    base = dfs[0][['Predictor']].copy()
    H = np.zeros((len(base), len(dfs)), dtype=np.float64)
    V = np.zeros((len(base), len(dfs)), dtype=np.float64)
    for i, df in enumerate(dfs):
        df = df.set_index('Predictor').reindex(base['Predictor']).reset_index()
        if df['h2'].isna().any() or df['sigma2'].isna().any():
            sys.exit(f'seed {args.seeds[i]} missing predictors present in seed {args.seeds[0]}')
        H[:, i] = df['h2'].values
        V[:, i] = df['sigma2'].values

    V_safe = np.clip(V, args.eps, None)
    w = 1.0 / V_safe
    h2_ivw = (H * w).sum(axis=1) / w.sum(axis=1)
    sigma2_simple = V.mean(axis=1)

    print(f'  IVW h^2:     mean={h2_ivw.mean():.3e}  med={np.median(h2_ivw):.3e}  '
          f'positive={int((h2_ivw > 0).sum())}/{len(h2_ivw)}')
    print(f'  simple s^2:  mean={sigma2_simple.mean():.3e}  med={np.median(sigma2_simple):.3e}')

    out_h2 = os.path.join(args.out_dir, 'h2_ivw.ind.her.pos.ens_ivw')
    out_sg = os.path.join(args.out_dir, 'sigma2_simple.ind.var.ens_simple')
    pd.DataFrame({'Predictor': base['Predictor'].values, 'h2': h2_ivw}) \
        .query('h2 > 0') \
        .to_csv(out_h2, sep='\t', index=False, header=False)
    pd.DataFrame({'Predictor': base['Predictor'].values, 'sigma2': sigma2_simple}) \
        .to_csv(out_sg, sep='\t', index=False, header=False)

    print(f'\n  -> {out_h2}')
    print(f'  -> {out_sg}')
    print('\nFeed both files to make_adaptive_prior.py to produce the final prior.')


if __name__ == '__main__':
    main()
