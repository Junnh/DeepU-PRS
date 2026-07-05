Trains one deep-ensemble member that maps per-SNP functional annotations to
(heritability prior h², predictive variance σ²). Run this with M different
seeds to produce the ensemble used downstream by make_ivw_prior.py and
make_adaptive_prior.py.

Architecture and loss are described in docs/architecture.md. Key choices:
  * Target  = log(S_j)  (chi-square variance-stabilising transform)
  * Loss    = Gaussian NLL on the log-transformed target
  * Model   = 4-layer MLP with separate (mean, log-variance) heads
  * LD agg  = sparse R² and R²² applied to (h², σ²) respectively
  * Stop    = relative early-stopping on raw MSE (1% threshold, patience 3)

CLI:
  python train.py \
      --summary path/to/<pheno>.train.summaries \
      --annot   path/to/annotations.csv \
      --ld_dir  path/to/ld_graphs/                \
      --maf     path/to/reference.frq \
      --out     ./out/seed_2023/ \
      --pca True --lr 0.001 --seed 2023

Required input formats:
  --summary  LDAK whitespace format with columns: Predictor A1 A2 Direction Stat n
  --annot    CSV with first column 'Predictor' and 65 (no-PCA) or 205 (PCA) feature columns
  --ld_dir   directory containing one file per chromosome:
               chrld_<c>.npy      (N_pairs x 3 array: [pred1, pred2, R^2])
               chr<c>_edge_index.npy (2 x N_pairs int64)
  --maf      reference allele-frequency file (plink .frq format)
"""
import argparse
import gc
import os
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch_geometric.data import Data

from pytorch_lamb import Lamb

from pytorchtools import RelativeEarlyStopping


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--summary', required=True, help='GWAS summary stats file (LDAK format)')
    p.add_argument('--annot',   required=True, help='Per-SNP annotation CSV (Predictor + features)')
    p.add_argument('--ld_dir',  required=True, help='Directory with chrld_<c>.npy and chr<c>_edge_index.npy')
    p.add_argument('--maf',     required=True, help='Reference MAF file (plink .frq format)')
    p.add_argument('--out',     required=True, help='Output directory for checkpoints + priors')
    p.add_argument('--pca',     type=lambda s: s.lower() == 'true', default=False,
                   help='True -> 205-feature input (functional + enformer PCA); False -> 65-feature input')
    p.add_argument('--lr',      type=float, default=0.001)
    p.add_argument('--seed',    type=int,   default=2023)
    p.add_argument('--epochs',  type=int,   default=40)
    p.add_argument('--patience', type=int,  default=3, help='Early-stop patience (relative 1%)')
    p.add_argument('--warmup',  type=int,   default=10, help='LR warmup epochs')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class FC(nn.Module):
    """Feed-forward annotation -> (h^2_prior, log sigma^2). See docs/architecture.md."""
    def __init__(self, pca: bool):
        super().__init__()
        self.pca = pca
        in_dim = 205 if pca else 65

        if pca:
            self.fc1 = nn.Linear(in_dim, 128); self.bn1 = nn.BatchNorm1d(128)
            self.fc3 = nn.Linear(128,   96);   self.bn3 = nn.BatchNorm1d(96)
            self.fc4 = nn.Linear(96,    64);   self.bn4 = nn.BatchNorm1d(64)

            self.avg_fc = nn.Linear(64, 32); self.avg_bn = nn.BatchNorm1d(32)
            self.var_fc = nn.Linear(64, 32); self.var_bn = nn.BatchNorm1d(32)
            self.avg = nn.Linear(32, 1)
            self.var = nn.Linear(32, 1)
        else:
            self.fc1 = nn.Linear(in_dim, 32); self.bn1 = nn.BatchNorm1d(32)
            self.fc3 = nn.Linear(32,     16); self.bn3 = nn.BatchNorm1d(16)
            self.fc4 = nn.Linear(16,      8); self.bn4 = nn.BatchNorm1d(8)
            self.avg_fc = nn.Linear(8, 8); self.avg_bn = nn.BatchNorm1d(8)
            self.var_fc = nn.Linear(8, 8); self.var_bn = nn.BatchNorm1d(8)
            self.avg = nn.Linear(8, 1)
            self.var = nn.Linear(8, 1)

        self.activation = nn.SiLU()
        self.softplus = nn.Softplus()
        # Log-Stat target lives near 1e-6 once n×h² is multiplied -> scale factor matches.
        self.scaling_factor = nn.Parameter(torch.tensor(1e-6))

    def forward(self, annot, freq_basis):
        h = self.activation(self.bn1(self.fc1(annot)))
        h = self.activation(self.bn3(self.fc3(h)))
        h = self.activation(self.bn4(self.fc4(h)))

        m = self.activation(self.avg_bn(self.avg_fc(h)))
        v = self.activation(self.var_bn(self.var_fc(h)))
        # h^2 is non-negative -> softplus, scaled by freq basis and learnable scale.
        h2     = freq_basis * self.softplus(self.avg(m)) * self.scaling_factor
        log_v  = self.var(v)
        return h2.reshape(-1, 1), log_v.reshape(-1, 1)


def init_weights(m: nn.Module):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
def nll_gaussian(y_mean: torch.Tensor, y_var: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Gaussian NLL on log-scale residual; variance clamped for numerical stability."""
    y_var = torch.clamp(y_var, min=1e-8, max=1e2)
    nll = 0.5 * torch.log(y_var) + 0.5 * ((y_mean - y_true) ** 2) / y_var
    return torch.mean(nll)


def compute_loss(aggregated_h2: torch.Tensor,
                 aggregated_var: torch.Tensor,
                 log_S: torch.Tensor,
                 n_per_snp: torch.Tensor,
                 valid_mask: torch.Tensor):
    """Loss in log-Stat space (target = log(S_j)).

    aggregated_h2:   (R^2) · h2,           shape (N, 1)
    aggregated_var:  (R^2)^2 · sigma^2,    shape (N, 1)
                     LD-aggregated aleatoric variance, used as the variance of
                     the Gaussian NLL on the log-Stat residual.
    log_S:           log(S_j),             shape (N,)
    n_per_snp:       per-SNP sample size,  shape (N,)
    valid_mask:      True for SNPs to include (S_j > 0)

    Returns (loss, raw_mse).
    """
    h2_flat   = aggregated_h2.reshape(-1)
    var_flat  = aggregated_var.reshape(-1)
    pred_log  = torch.log1p(n_per_snp * torch.clamp(h2_flat, min=0.0))

    loss = nll_gaussian(pred_log[valid_mask], var_flat[valid_mask], log_S[valid_mask])
    with torch.no_grad():
        raw_mse = torch.mean((pred_log[valid_mask] - log_S[valid_mask]) ** 2).item()
    return loss, raw_mse


# ---------------------------------------------------------------------------
# LD-aggregated forward (sparse mm with CSR)
# ---------------------------------------------------------------------------
def ld_aggregated_forward(model, dataset, chr_num: int, device):
    """Single-chromosome forward: produce LD-aggregated (h2, var) targets.

    dataset[chr_num] is a dict with keys: data (Data), annot (tensor), freq_bs (tensor),
                                          csr (crow, col, vals, size).
    """
    entry  = dataset[chr_num]
    data   = entry['data']
    annot  = entry['annot'].to(device)
    fb     = entry['freq_bs'].to(device)
    crow, col, vals, size = entry['csr']
    crow = crow.to(device); col = col.to(device); vals = vals.to(device)

    h2, log_v = model.FC_net(annot, fb)
    sigma2 = torch.exp(log_v)

    # `vals` already contains R^2 (chrld[:, 2] is R^2 — see build_chr_dataset
    # docstring). So R2 is the R^2 sparse matrix and R4 = (R^2)^2 is its
    # entry-wise square, used to sum aleatoric variance in quadrature.
    R2 = torch.sparse_csr_tensor(crow, col, vals,         (size, size))
    R4 = torch.sparse_csr_tensor(crow, col, vals * vals,  (size, size))
    aggregated_h2  = torch.sparse.mm(R2, h2)        # R^2 · h^2
    aggregated_var = torch.sparse.mm(R4, sigma2)    # (R^2)^2 · sigma^2

    return aggregated_h2, data.y.to(device), aggregated_var, data.n.to(device), data.mask.to(device)


class Pass(nn.Module):
    """Lightweight wrapper to make FC_net + dataset addressable by chr_num."""
    def __init__(self, FC_net, dataset):
        super().__init__()
        self.FC_net = FC_net
        self.dataset = dataset


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def build_chr_dataset(annot: pd.DataFrame, summaries: pd.DataFrame, ref_maf: pd.DataFrame,
                      chr_list, ld_dir: str):
    """For each chromosome build the (Data, annot, freq_basis, CSR) bundle.

    Note: this routine is illustrative; the production version caches CSRs to disk
    per chromosome to avoid rebuilding them on every run.
    """
    annot_g = annot.groupby('chr')
    summ_g  = summaries.groupby('chr')
    maf_g   = ref_maf.groupby('CHR')

    dataset = {}
    for c in chr_list:
        sc = str(c)
        a_c = annot_g.get_group(sc).reset_index(drop=True)
        s_c = summ_g .get_group(sc).reset_index(drop=True)
        m_c = maf_g  .get_group(int(c))
        m_c = m_c[m_c['SNP'].isin(s_c['Predictor'].values)]

        # LD: pairs (pred1, pred2, R^2)
        chrld = np.load(os.path.join(ld_dir, f'chrld_{c}.npy'), allow_pickle=True)
        valid = s_c['Predictor'].values
        keep = (pd.Series(chrld[:, 0]).isin(valid) & pd.Series(chrld[:, 1]).isin(valid)).values
        edge_attr  = torch.tensor(chrld[keep][:, 2].astype(np.float32), dtype=torch.float32)
        edge_index = torch.tensor(np.load(os.path.join(ld_dir, f'chr{c}_edge_index.npy')).T,
                                  dtype=torch.long)

        # target: log(Stat), per-SNP n, valid mask
        log_S    = torch.tensor(np.log(s_c['Stat'].values.astype(np.float32)), dtype=torch.float32)
        n_per    = torch.tensor(s_c['n'   ].values.astype(np.float32), dtype=torch.float32)
        mask_arr = torch.tensor(s_c['valid_mask'].values.astype(bool), dtype=torch.bool)

        annot_t = torch.tensor(a_c.drop(['Predictor', 'chr'], axis=1).values.astype(np.float32),
                               dtype=torch.float32)
        maf = torch.tensor(m_c['MAF'].values.astype(np.float32), dtype=torch.float32)
        freq_basis = ((maf * (1 - maf)) ** 0.75).reshape(-1, 1)

        d = Data(num_nodes=len(s_c), edge_index=edge_index, edge_attr=edge_attr,
                 y=log_S, n=n_per, mask=mask_arr)

        # Build CSR for sparse mm
        size = len(s_c)
        i3 = torch.cat([edge_index, torch.flip(edge_index, dims=[0]),
                        torch.arange(size).reshape(1, -1).repeat(2, 1)], dim=1).contiguous()
        v3 = torch.cat([edge_attr, edge_attr, torch.ones(size)])
        coo = torch.sparse_coo_tensor(i3, v3, (size, size)).coalesce()
        csr = coo.to_sparse_csr()
        crow = csr.crow_indices().to(torch.int32).contiguous()
        col  = csr.col_indices ().to(torch.int32).contiguous()
        vals = csr.values().contiguous()

        dataset[c] = {'data': d, 'annot': annot_t, 'freq_bs': freq_basis,
                       'csr': (crow, col, vals, size)}
        del chrld
        gc.collect()
    return dataset


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def adjust_lr(optimizer, epoch, warmup, base_lr):
    if epoch < warmup:
        for pg in optimizer.param_groups:
            pg['lr'] = base_lr * float(epoch + 1) / warmup
    else:
        for pg in optimizer.param_groups:
            pg['lr'] = base_lr


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device={device}  pca={args.pca}  seed={args.seed}  lr={args.lr}")

    # ------------------------- Load data ---------------------------
    summaries = pd.read_csv(args.summary, sep=r"\s+", engine='c')
    summaries['valid_mask'] = summaries['Stat'] > 0
    n_inv = int((~summaries['valid_mask']).sum())
    print(f"masked invalid SNPs: {n_inv:,} / {len(summaries):,}")
    summaries.loc[summaries['Stat'] <= 0, 'Stat'] = 1e-6
    summaries['chr'] = summaries['Predictor'].str.split(':').str[0]

    annot   = pd.read_csv(args.annot, engine='c')
    annot['chr'] = annot['Predictor'].str.split(':').str[0]
    ref_maf = pd.read_csv(args.maf, sep=r"\s+", engine='c')

    chr_list = np.arange(1, 23)
    print("Building per-chromosome datasets...")
    dataset = build_chr_dataset(annot, summaries, ref_maf, chr_list, args.ld_dir)

    # ------------------------- Model & optim -----------------------
    FC_net = FC(pca=args.pca)
    FC_net.apply(init_weights)
    model = Pass(FC_net, dataset).to(device)
    optimizer = Lamb(model.parameters(), lr=args.lr, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, threshold=1e-2, min_lr=1e-6)
    early_stopping = RelativeEarlyStopping(patience=args.patience, min_rel_improve=0.01, verbose=True)

    # ------------------------- Training loop -----------------------
    chr_train = np.arange(1, 23)
    for e in range(args.epochs):
        t0 = time.time()
        running_loss = 0.0
        running_mse  = 0.0

        if e < args.warmup:
            adjust_lr(optimizer, e, args.warmup, args.lr)
        model.train()
        np.random.shuffle(chr_train)

        for c in chr_train:
            optimizer.zero_grad()
            agg_h2, log_S, agg_var, n_per, mask = ld_aggregated_forward(model, dataset, c, device)
            loss, raw_mse = compute_loss(agg_h2, agg_var, log_S, n_per, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            running_loss += loss.item()
            running_mse  += raw_mse

        epoch_loss = running_loss / len(chr_train)
        epoch_mse  = running_mse  / len(chr_train)
        print(f"epoch {e:3d}  loss={epoch_loss:.4e}  raw_mse={epoch_mse:.4e}  "
              f"time={time.time() - t0:.1f}s")

        torch.save(model.state_dict(),
                   os.path.join(args.out, f"checkpoint_e{e:03d}_seed{args.seed}.pt"))
        early_stopping(epoch_mse, model, epoch=e)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {e}; best epoch {early_stopping.best_epoch}.")
            break
        if e >= args.warmup:
            scheduler.step(epoch_mse)

    # ------------------------- Final inference ---------------------
    # Use the best-epoch checkpoint to produce per-SNP (h2, sigma^2) outputs.
    best_e = early_stopping.best_epoch if early_stopping.best_epoch >= 0 else e
    best_ckpt = os.path.join(args.out, f"checkpoint_e{best_e:03d}_seed{args.seed}.pt")
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    model.eval()

    all_pred_h2, all_pred_var, all_predictors = [], [], []
    with torch.no_grad():
        for c in chr_list:
            entry = dataset[c]
            annot_t = entry['annot'].to(device)
            fb      = entry['freq_bs'].to(device)
            h2, log_v = model.FC_net(annot_t, fb)
            all_pred_h2.append(h2.cpu().numpy().reshape(-1))
            all_pred_var.append(torch.exp(log_v).cpu().numpy().reshape(-1))

    # Build predictor index from summaries (the order build_chr_dataset uses).
    for c in chr_list:
        sc = str(c)
        s_c = summaries[summaries['chr'] == sc].reset_index(drop=True)
        all_predictors.append(s_c['Predictor'].values)
    predictors = np.concatenate(all_predictors)
    h2_out  = np.concatenate(all_pred_h2)
    sig_out = np.concatenate(all_pred_var)

    out_csv_h2  = os.path.join(args.out, f"per_snp_h2_seed{args.seed}.tsv")
    out_csv_sig = os.path.join(args.out, f"per_snp_sigma2_seed{args.seed}.tsv")
    pd.DataFrame({'Predictor': predictors, 'h2': h2_out}).to_csv(
        out_csv_h2, sep='\t', index=False, header=False)
    pd.DataFrame({'Predictor': predictors, 'sigma2': sig_out}).to_csv(
        out_csv_sig, sep='\t', index=False, header=False)
    print(f"\nWrote per-SNP h2     : {out_csv_h2}")
    print(f"Wrote per-SNP sigma^2: {out_csv_sig}")
    print("\nNext steps:")
    print("  1. Train other seeds.")
    print("  2. Aggregate (IVW for h2, simple-mean for sigma2) across seeds.")
    print("  3. Apply make_adaptive_prior.py to produce the LDAK .ind.her.pos.adaptive file.")


if __name__ == "__main__":
    main()
