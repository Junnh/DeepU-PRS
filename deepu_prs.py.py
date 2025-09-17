#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DeepU-PRS
-----------------------------------

Pipeline
1) Load per-SNP annotations (+ optional PCA/enformer features) and summary stats.
2) For each chromosome (1..22):
   - Load LD edges (edge_index, edge_attr=r^2) & build torch_geometric Data.
   - Build design matrices: annotation tensor and MAF-based frequency basis.
   - FCNet computes per-SNP prior mean/log-variance.
   - Aggregate to per-SNP chi-square expectation via sparse LD (CPU sparse.mm).
3) Train with Gaussian NLL on ((Stat-1)/n) with early stopping.
4) Export per-SNP prior mean & logvar + cohort files (full/positive/snplist).

Data layout (fill with your own files)
<base_dir>/
├─ annotations/annotation.csv                # columns: Predictor, feat1..featK
├─ summaries/train.summaries                 # columns: Predictor, Stat, n, ...
├─ maf/plink.frq                             # columns: CHR, SNP, MAF, ...
└─ ld/
   ├─ edge_index/chr{1..22}_edge_index.npy   # int indices; shape (2,E) or (E,2)
   └─ ld_triplets/chrld_{1..22}.npy          # rows: [SNP_A, SNP_B, r2]

Usage (example)
python deepu_prs_refactored.py \
  --file_path HDL \
  --pca true \
  --lr 1e-3 \
  --seed 2025 \
  --ver v1 \
  --base_dir [] \
  --output_dir []
"""
from __future__ import annotations

import argparse
import gc
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data

# ------------------------
# Config
# ------------------------

@dataclass
class TrainConfig:
    file_path: str
    pca: bool
    lr: float
    seed: int
    ver: str | None
    base_dir: Path
    output_dir: Path
    epochs: int = 70
    patience: int = 6
    warmup_epochs: int = 5
    weight_decay: float = 1e-3
    clip_grad_norm: float = 1.0
    min_var: float = 1e-8
    max_var: float = 1e2
    device: str = "cpu"  

class EarlyStopping:
    """Minimal early stopping on a scalar metric (lower is better)."""
    def __init__(self, patience: int = 6, delta: float = 0.01, checkpoint_path: Path | None = None):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.checkpoint_path = checkpoint_path

    def __call__(self, val_loss: float, model: nn.Module):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self._save(model)
            self.counter = 0
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self._save(model)
            self.counter = 0

    def _save(self, model: nn.Module):
        if self.checkpoint_path is not None:
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), self.checkpoint_path.as_posix())

# ------------------------
# Optimizer: Minimal LAMB (no external deps)
# ------------------------

class Lamb(torch.optim.Optimizer):
    """Lightweight LAMB optimizer (no bias correction to keep it minimal)."""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']
            lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                m = state['exp_avg']
                v = state['exp_avg_sq']
                m.mul_(beta1).add_(g, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                update = m / (v.sqrt().add_(eps))
                if wd != 0:
                    update = update.add(p, alpha=wd)
                w_norm = torch.linalg.norm(p)
                u_norm = torch.linalg.norm(update)
                trust = (w_norm / u_norm) if (w_norm > 0 and u_norm > 0) else torch.tensor(1.0, device=p.device)
                p.add_(update, alpha=-lr * trust)
        return None

# ------------------------
# Model
# ------------------------

class FCNet(nn.Module):
    """Annotation -> per-SNP prior mean/logvar."""
    def __init__(self, in_dim_pca: int = 205, in_dim_basic: int = 65, pca: bool = True):
        super().__init__()
        self.pca = pca
        self.softplus = nn.Softplus()
        self.scaling = nn.Parameter(torch.tensor(1e-4))  # keep fixed init scale as requested
        if self.pca:
            self.fc1 = nn.Linear(in_dim_pca, 128); self.bn1 = nn.BatchNorm1d(128)
            self.fc2 = nn.Linear(128, 96);        self.bn2 = nn.BatchNorm1d(96)
            self.fc3 = nn.Linear(96, 64);         self.bn3 = nn.BatchNorm1d(64)
            self.avg_fc = nn.Linear(64, 32);      self.avg_bn = nn.BatchNorm1d(32)
            self.var_fc = nn.Linear(64, 32);      self.var_bn = nn.BatchNorm1d(32)
            self.avg = nn.Linear(32, 1)
            self.var = nn.Linear(32, 1)
        else:
            self.fc1 = nn.Linear(in_dim_basic, 64); self.bn1 = nn.BatchNorm1d(64)
            self.fc2 = nn.Linear(64, 32);           self.bn2 = nn.BatchNorm1d(32)
            self.fc3 = nn.Linear(32, 16);           self.bn3 = nn.BatchNorm1d(16)
            self.avg_fc = nn.Linear(16, 8);         self.avg_bn = nn.BatchNorm1d(8)
            self.var_fc = nn.Linear(16, 8);         self.var_bn = nn.BatchNorm1d(8)
            self.avg = nn.Linear(8, 1)
            self.var = nn.Linear(8, 1)
        self.act = nn.SiLU()
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, annot: torch.Tensor, freq_basis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.act(self.bn1(self.fc1(annot)))
        x = self.act(self.bn2(self.fc2(x)))
        x = self.act(self.bn3(self.fc3(x)))
        m = self.act(self.avg_bn(self.avg_fc(x)))
        v = self.act(self.var_bn(self.var_fc(x)))
        mean = freq_basis * self.softplus(self.avg(m)) * self.scaling
        logvar = self.var(v) + 2 * torch.log(freq_basis + 1e-12) + 2 * torch.log(self.scaling.abs() + 1e-12)
        return mean.view(-1, 1), logvar.view(-1, 1)

class PassModel(nn.Module):
    """Wrap FCNet and aggregate via LD (CPU sparse ops to save GPU memory)."""
    def __init__(self, fc_net: FCNet, dataset):
        super().__init__()
        self.fc_net = fc_net
        self.dataset = dataset

    def forward(self, chr_num: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        data, annot, freq_bs = self.dataset[chr_num]  # all CPU tensors
        prior, logvar = self.fc_net(annot, freq_bs)   # CPU forward to keep gradients intact

        # Build symmetric LD with self-loops (CPU)
        n = data.x.shape[0]
        diag = torch.arange(n, device=torch.device("cpu")).view(1, -1).repeat(2, 1)
        one = torch.ones(n, device=torch.device("cpu"))
        edge_index_cpu = data.edge_index.cpu()
        edge_attr_cpu = data.edge_attr.cpu()
        i3 = torch.cat([edge_index_cpu, torch.flip(edge_index_cpu, dims=[0]), diag], dim=1)
        v3 = torch.cat([edge_attr_cpu, edge_attr_cpu, one])

        s3 = torch.sparse_coo_tensor(i3, v3, (n, n))
        s3_sq = torch.sparse_coo_tensor(i3, v3 * v3, (n, n))

        y_mean = torch.sparse.mm(s3, prior).view(-1)
        y_var = torch.sparse.mm(s3_sq, (logvar.exp())).view(-1)
        y_true = data.y.view(-1)
        return y_mean, y_true, y_var

# ------------------------
# IO Helpers
# ------------------------

def _add_chr_column(df: pd.DataFrame) -> pd.DataFrame:
    if "chr" not in df.columns:
        df = df.copy()
        df["chr"] = df["Predictor"].astype(str).str.split(":").str[0]
    return df

def load_inputs(cfg: TrainConfig, logger: logging.Logger):
    """Load annotations, summaries, ref MAF, LD edges per chromosome (fixed paths)."""
    annot_path = cfg.base_dir / "annotations" / "annotation.csv"
    summ_path  = cfg.base_dir / "summaries"  / "train.summaries"
    ref_maf_path = cfg.base_dir / "maf" / "plink.frq"

    logger.info(f"Loading annot: {annot_path}")
    annot = pd.read_csv(annot_path, engine="c")
    logger.info(f"Loading summaries: {summ_path}")
    summaries = pd.read_csv(summ_path, delim_whitespace=True)
    logger.info(f"Loading ref MAF: {ref_maf_path}")
    ref_maf = pd.read_csv(ref_maf_path, delim_whitespace=True)

    # Guard: Stat <= 0 -> 1e-6
    zero_idx = summaries.loc[summaries["Stat"] <= 0].index
    if len(zero_idx) > 0:
        summaries.loc[zero_idx, "Stat"] = 1e-6

    annot = _add_chr_column(annot)
    summaries = _add_chr_column(summaries)

    chr_list = list(range(1, 23))
    per_chr: Dict[int, Tuple[Data, torch.Tensor, torch.Tensor]] = {}

    def edge_index_path(chr_num: int) -> Path:
        # Expect: <base_dir>/ld/edge_index/chr{chr}_edge_index.npy
        return cfg.base_dir / "ld" / "edge_index" / f"chr{chr_num}_edge_index.npy"

    def ld_triplet_path(chr_num: int) -> Path:
        # Expect: <base_dir>/ld/ld_triplets/chrld_{chr}.npy
        return cfg.base_dir / "ld" / "ld_triplets" / f"chrld_{chr_num}.npy"

    for chr_num in chr_list:
        a_df = annot.loc[annot["chr"] == str(chr_num)].copy()
        s_df = summaries.loc[summaries["chr"] == str(chr_num)].copy()

        maf_chr = ref_maf.loc[ref_maf["CHR"] == int(chr_num)].copy()
        maf_chr = maf_chr.loc[maf_chr["SNP"].isin(s_df["Predictor"].unique())]

        annot_vals = torch.tensor(a_df.drop(columns=["Predictor", "chr"]).values, dtype=torch.float32)
        maf = torch.tensor(maf_chr["MAF"].values, dtype=torch.float32)
        maf_bias = ((maf * (1 - maf)) ** 0.75).view(-1, 1)

        edge_idx_np = np.load(edge_index_path(chr_num))
        if edge_idx_np.shape[0] != 2:
            edge_idx_np = edge_idx_np.T
        edge_index = torch.tensor(edge_idx_np, dtype=torch.long)

        ld_triplet = np.load(ld_triplet_path(chr_num), allow_pickle=True)
        mask = (
            pd.DataFrame(ld_triplet[:, 0]).isin(s_df["Predictor"].values).values
            & pd.DataFrame(ld_triplet[:, 1]).isin(s_df["Predictor"].values).values
        ).reshape(-1)
        edge_attr = torch.tensor(ld_triplet[mask][:, 2].astype(np.float32), dtype=torch.float32)

        y = torch.tensor(((s_df["Stat"] - 1) / s_df["n"]).values, dtype=torch.float32)

        n_nodes = a_df["Predictor"].nunique()
        data = Data(x=torch.arange(n_nodes), edge_index=edge_index, edge_attr=edge_attr, y=y)

        per_chr[chr_num] = (data, annot_vals, maf_bias)

        del annot_vals, maf, maf_bias, edge_index, edge_attr, y, data
        gc.collect()

    return per_chr, annot, summaries

# ------------------------
# Loss / LR helpers
# ------------------------

def gaussian_nll(y_mean: torch.Tensor, y_var: torch.Tensor, y_true: torch.Tensor,
                 min_var: float, max_var: float) -> torch.Tensor:
    y_var = torch.clamp(y_var, min=min_var, max=max_var)
    return torch.mean(0.5 * torch.log(y_var + 1e-12) + 0.5 * (y_mean - y_true) ** 2 / (y_var + 1e-12))

def warmup_lr(optimizer: torch.optim.Optimizer, epoch: int, warmup_epochs: int, base_lr: float):
    lr = base_lr * float(epoch + 1) / max(1, warmup_epochs) if epoch < warmup_epochs else base_lr
    for pg in optimizer.param_groups:
        pg["lr"] = lr

# ------------------------
# Train
# ------------------------

def train(cfg: TrainConfig, logger: logging.Logger):
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)

    dataset, annot_df, summaries = load_inputs(cfg, logger)

    in_dim = annot_df.drop(columns=["Predictor", "chr"]).shape[1]
    fc_net = FCNet(in_dim_pca=in_dim, in_dim_basic=in_dim, pca=cfg.pca).to("cpu")  

    dataset_cpu = {chr_num: (data, annot_vals, maf_bias)
                   for chr_num, (data, annot_vals, maf_bias) in dataset.items()}

    model = PassModel(fc_net, dataset_cpu).to("cpu")

    optimizer = Lamb(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, threshold=1e-2, min_lr=1e-6)

    ckpt = cfg.output_dir / f"f{cfg.ver}_best.pt"
    early_stop = EarlyStopping(patience=cfg.patience, delta=0.01, checkpoint_path=ckpt)

    chr_train = list(range(1, 23))

    for epoch in range(cfg.epochs):
        model.train()
        warmup_lr(optimizer, epoch, cfg.warmup_epochs, cfg.lr)
        np.random.shuffle(chr_train)

        train_loss = 0.0
        for chr_num in chr_train:
            optimizer.zero_grad()
            y_mean, y_true, y_var = model(chr_num)
            loss = gaussian_nll(y_mean, y_var, y_true, cfg.min_var, cfg.max_var)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm)
            optimizer.step()
            train_loss += loss.item()

        with torch.no_grad():
            model.eval()
            val_loss = 0.0
            for chr_num in chr_train:
                y_mean, y_true, y_var = model(chr_num)
                val_loss += gaussian_nll(y_mean, y_var, y_true, cfg.min_var, cfg.max_var).item()

        scheduler.step(val_loss)
        early_stop(val_loss, model)

        logger.info(f"Epoch {epoch:03d} | train={train_loss:.4f} val={val_loss:.4f} lr={optimizer.param_groups[0]['lr']:.2e}")

        if early_stop.early_stop:
            logger.info("Early stopping triggered.")
            break

    # Load best
    model.load_state_dict(torch.load(ckpt.as_posix(), map_location="cpu"))
    model.eval()

    # Export per-SNP prior mean/logvar
    all_prior, all_logvar = [], []
    with torch.no_grad():
        for chr_num in chr_train:
            _, annot_vals, maf_bias = dataset_cpu[chr_num]
            pr, lv = fc_net(annot_vals, maf_bias)
            all_prior.append(pr.cpu()); all_logvar.append(lv.cpu())

    prior = torch.cat(all_prior).squeeze(1).numpy()
    logvar = torch.cat(all_logvar).squeeze(1).numpy()

    out_base = cfg.output_dir / ("enformer" if cfg.pca else "deep_imp0")
    out_base.mkdir(parents=True, exist_ok=True)

    deep_her = pd.DataFrame({"Predictor": summaries["Predictor"].values, "Heritability": prior})
    suffix = f"f{cfg.ver}_lr{cfg.lr}_seed{cfg.seed}"

    deep_her[["Predictor", "Heritability"]].to_csv((out_base / f"{suffix}.noamb.ind.her").as_posix(),
                                                   sep="\t", index=False, header=False)

    pos = deep_her.loc[deep_her["Heritability"] > 0]
    pos[["Predictor", "Heritability"]].to_csv((out_base / f"{suffix}.noamb.ind.her.pos").as_posix(),
                                              sep="\t", index=False, header=False)

    summaries.loc[summaries["Predictor"].isin(pos["Predictor"])]["Predictor"].to_csv(
        (out_base / f"{suffix}.noamb_snps.txt").as_posix(), sep="\t", index=False, header=False
    )

    deep_her_log = deep_her.copy()
    deep_her_log["logvar"] = logvar
    deep_her_log.to_csv((out_base / f"{suffix}.noamb.ind.her.logvar.csv").as_posix(), index=False)

    logger.info("Export complete.")

# ------------------------
# CLI
# ------------------------

def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--file_path", type=str, required=True, help="Trait folder name (used for output subfolder)")
    p.add_argument("--pca", type=lambda s: str(s).lower() in {"1","true","t","yes","y"}, default=False)
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--ver", type=str, default="v1")

    p.add_argument("--base_dir", type=Path, default=Path("."))
    p.add_argument("--output_dir", type=Path, default=None, help="Defaults to <base_dir>/<file_path>")

    p.add_argument("--epochs", type=int, default=70)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--clip_grad_norm", type=float, default=1.0)
    p.add_argument("--min_var", type=float, default=1e-8)
    p.add_argument("--max_var", type=float, default=1e2)
    p.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG","INFO","WARNING","ERROR","CRITICAL"])

    args = p.parse_args()
    output_dir = args.output_dir or (args.base_dir / f"{args.file_path}")
    return TrainConfig(
        file_path=args.file_path,
        pca=args.pca,
        lr=args.lr,
        seed=args.seed,
        ver=args.ver,
        base_dir=args.base_dir,
        output_dir=output_dir,
        epochs=args.epochs,
        patience=args.patience,
        warmup_epochs=args.warmup_epochs,
        weight_decay=args.weight_decay,
        clip_grad_norm=args.clip_grad_norm,
        min_var=args.min_var,
        max_var=args.max_var,
    )

def main():
    cfg = parse_args()
    logging.basicConfig(level=getattr(logging, "INFO"), format="[%(asctime)s] %(levelname)s: %(message)s")
    logger = logging.getLogger("deepu_prs"); logger.setLevel(logging.INFO)
    logger.info("Device: CPU (LD ops on CPU to preserve grads)")
    logger.info(str(cfg))
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    train(cfg, logger)

if __name__ == "__main__":
    main()
