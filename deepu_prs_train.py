#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DeepU-PRS training script (cleaned & GitHub-ready)

- Uncertainty-aware FC network that predicts per-SNP prior mean & log-variance
- Chromosome-wise sparse LD application
- Warmup, ReduceLROnPlateau scheduler, early stopping
- Deterministic seeds, logging, and argumentized paths

Original ideas & logic adapted from user's script.
"""

from __future__ import annotations
import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List

import gc
import psutil
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data
from pytorch_lamb import Lamb

# -----------------------------
# Logging
# -----------------------------
def setup_logger(level: str = "INFO") -> None:
    logging.basicConfig(
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        level=getattr(logging, level.upper(), logging.INFO),
        stream=sys.stdout,
    )

# -----------------------------
# EarlyStopping
# -----------------------------
class EarlyStopping:
    def __init__(self, patience: int = 6, delta: float = 0.0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> None:
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.counter = 0
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

# -----------------------------
# Args / Config
# -----------------------------
@dataclass
class Config:
    file_path: str
    biomarker: str
    lr: float
    r2_coverage: str = "cut_0.01/cross_chr"
    pca: bool = False
    sim: bool = False
    her: str | None = None
    causal: str | None = None
    seed: int = 2025
    ver: str | None = None

    # paths (argumentize all previous hard-coding)
    path_ref_maf: str = "./ref/plink.frq"
    path_annot_pca: str = "./annot/annot_all_mmscaled.csv"
    path_annot_default: str = "./annot/annot_imp_v2.csv"
    path_summaries_train: str = "./summaries/{file_path}/neale.train.summaries"
    path_ld_edge_attr: str = "./ld/{r2_coverage}/chrld_{chrom}.npy"
    path_edge_index: str = "./edge/{r2_coverage}/chr{chrom}_edge_index.npy"
    # outputs
    out_dir: str = "./outputs"

    # training
    epochs: int = 70
    patience: int = 6
    warmup_epochs: int = 5

def parse_args() -> Config:
    p = argparse.ArgumentParser()
    p.add_argument("--file_path", type=str, required=True)
    p.add_argument("--biomarker", type=str, required=True)
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--r2_coverage", type=str, default="cut_0.01/cross_chr")
    p.add_argument("--pca", action="store_true")
    p.add_argument("--sim", action="store_true")
    p.add_argument("--her", type=str, default=None)
    p.add_argument("--causal", type=str, default=None)
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--ver", type=str, default=None)

    # optional: override paths
    p.add_argument("--path_ref_maf", type=str, default="./ref/plink.frq")
    p.add_argument("--path_annot_pca", type=str, default="./annot/annot_all_mmscaled.csv")
    p.add_argument("--path_annot_default", type=str, default="./annot/annot_imp_v2.csv")
    p.add_argument("--path_summaries_train", type=str, default="./summaries/{file_path}/neale.train.summaries")
    p.add_argument("--path_ld_edge_attr", type=str, default="./ld/{r2_coverage}/chrld_{chrom}.npy")
    p.add_argument("--path_edge_index", type=str, default="./edge/{r2_coverage}/chr{chrom}_edge_index.npy")
    p.add_argument("--out_dir", type=str, default="./outputs")

    # training
    p.add_argument("--epochs", type=int, default=70)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--warmup_epochs", type=int, default=5)

    args = p.parse_args()
    cfg = Config(**vars(args))
    return cfg

# -----------------------------
# Utils
# -----------------------------
def set_deterministic(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)

def load_ref_maf(path_ref_maf: str) -> pd.DataFrame:
    return pd.read_csv(path_ref_maf, delim_whitespace=True)

def load_annotation(cfg: Config) -> pd.DataFrame:
    if cfg.pca:
        path = cfg.path_annot_pca
    else:
        path = cfg.path_annot_default
    annot = pd.read_csv(path, engine="c")
    return annot

def load_summaries(cfg: Config) -> pd.DataFrame:
    if cfg.sim:
        sim_path = f"./simulation/simul.{cfg.her}.{cfg.causal}.train.summaries"
        summaries = pd.read_csv(sim_path, delim_whitespace=True)
    else:
        path = cfg.path_summaries_train.format(file_path=cfg.file_path)
        summaries = pd.read_csv(path, delim_whitespace=True)

    # avoid zeros
    zero_idx = summaries[summaries["Stat"] <= 0].index
    summaries.loc[zero_idx, "Stat"] = 1e-6
    return summaries

def add_chr_column(df: pd.DataFrame, col: str = "Predictor") -> pd.DataFrame:
    out = df.copy()
    out["chr"] = out[col].astype(str).str.split(":").str[0]
    return out

def ld_edge_attr_path(cfg: Config, chrom: int) -> str:
    return cfg.path_ld_edge_attr.format(r2_coverage=cfg.r2_coverage, chrom=chrom)

def edge_index_path(cfg: Config, chrom: int) -> str:
    return cfg.path_edge_index.format(r2_coverage=cfg.r2_coverage, chrom=chrom)

def gaussian_nll(y_mean: torch.Tensor, y_var: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    y_var = torch.clamp(y_var, min=1e-10, max=1e2)
    return torch.mean(0.5 * torch.log(y_var) + 0.5 * ((y_mean - y_true) ** 2) / y_var)

# -----------------------------
# Model
# -----------------------------
class FCUncertainty(nn.Module):
    """
    FC-205 or FC-65 with aleatoric uncertainty
    """
    def __init__(self, pca: bool):
        super().__init__()
        if pca:
            in_dim = 205
            self.fc1 = nn.Linear(in_dim, 128)
            self.bn1 = nn.BatchNorm1d(128)
            self.fc2 = nn.Linear(128, 96)
            self.bn2 = nn.BatchNorm1d(96)
            self.fc3 = nn.Linear(96, 64)
            self.bn3 = nn.BatchNorm1d(64)

            self.avg_fc = nn.Linear(64, 32)
            self.avg_bn = nn.BatchNorm1d(32)

            self.var_fc = nn.Linear(64, 32)
            self.var_bn = nn.BatchNorm1d(32)

            self.avg_head = nn.Linear(32, 1)
            self.var_head = nn.Linear(32, 1)
        else:
            in_dim = 65
            self.fc1 = nn.Linear(in_dim, 32)
            self.bn1 = nn.BatchNorm1d(32)
            self.fc2 = nn.Linear(32, 16)
            self.bn2 = nn.BatchNorm1d(16)
            self.fc3 = nn.Linear(16, 16)
            self.bn3 = nn.BatchNorm1d(16)

            self.avg_fc = nn.Linear(16, 8)
            self.avg_bn = nn.BatchNorm1d(8)
            self.var_fc = nn.Linear(16, 8)
            self.var_bn = nn.BatchNorm1d(8)
            self.avg_head = nn.Linear(8, 1)
            self.var_head = nn.Linear(8, 1)

        self.act = nn.SiLU()
        self.softplus = nn.Softplus()
        # learnable scale like original code
        self.scaling_factor = nn.Parameter(torch.tensor(0.0001))

        # init
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, freq_basis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # trunk
        x = self.act(self.bn1(self.fc1(x)))
        x = self.act(self.bn2(self.fc2(x)))
        x = self.act(self.bn3(self.fc3(x)))

        # average & variance heads
        mean = self.act(self.avg_bn(self.avg_fc(x)))
        logvar = self.act(self.var_bn(self.var_fc(x)))

        # scale with maf basis and learnable scaling (preserving original logic)
        mean = freq_basis * self.softplus(self.avg_head(mean)) * self.scaling_factor
        logvar = self.var_head(logvar) + 2 * torch.log(freq_basis) + 2 * torch.log(self.scaling_factor)

        mean = mean.reshape(-1, 1)
        logvar = logvar.reshape(-1, 1)
        return mean, logvar

# -----------------------------
# Chromosome Dataset (in-memory tuples, like original)
# -----------------------------
@dataclass
class ChrPack:
    data: Data
    annot: torch.Tensor
    freq_basis: torch.Tensor
    target_y: torch.Tensor  # (Stat - 1)/n

def build_chr_packs(
    cfg: Config, annot_df: pd.DataFrame, summaries_df: pd.DataFrame, ref_maf_df: pd.DataFrame, device: torch.device
) -> Dict[int, ChrPack]:
    packs: Dict[int, ChrPack] = {}
    chr_list = list(range(1, 23))

    for chrom in chr_list:
        # slice
        annot_chr = annot_df[annot_df["chr"] == str(chrom)]
        sums_chr = summaries_df[summaries_df["chr"] == str(chrom)]
        maf_chr = ref_maf_df[ref_maf_df["CHR"] == chrom]
        maf_chr = maf_chr[maf_chr["SNP"].isin(sums_chr["Predictor"].unique())]

        # tensors
        y = torch.tensor(((sums_chr["Stat"] - 1) / sums_chr["n"]).values, dtype=torch.float32)
        annot_vals = torch.tensor(annot_chr.drop(columns=["Predictor", "chr"]).values, dtype=torch.float32)

        maf = torch.tensor(maf_chr["MAF"].values, dtype=torch.float32)
        freq_basis = ((maf * (1 - maf)) ** 0.75).reshape(-1, 1)

        # edges & weights
        edge_attr_np = np.load(ld_edge_attr_path(cfg, chrom), allow_pickle=True)
        # filter to keep only entries where both nodes exist in our summaries set is already done offline in your pipeline;
        edge_attr = torch.tensor(edge_attr_np[:, 2].astype(np.float32), dtype=torch.float32)

        edge_index_np = np.load(edge_index_path(cfg, chrom))
        if edge_index_np.ndim == 2 and edge_index_np.shape[0] != 2:
            edge_index_np = edge_index_np.T
        edge_index = torch.tensor(edge_index_np, dtype=torch.long)

        # build pyg Data (nodes are implicit via x indices)
        n_nodes = annot_chr["Predictor"].nunique()
        data = Data(x=torch.arange(n_nodes), edge_index=edge_index, edge_attr=edge_attr, y=y)

        packs[chrom] = ChrPack(
            data=data,
            annot=annot_vals.to(device),
            freq_basis=freq_basis.to(device),
            target_y=y,  # kept on CPU; moved as needed
        )

        # memory hygiene
        del maf, y, annot_vals, edge_attr_np, edge_index_np
        gc.collect()

    return packs

# -----------------------------
# Sparse apply (LD * mean / variance)
# -----------------------------
def apply_sparse_ld(data: Data, prior_mean: torch.Tensor, prior_logvar: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build symmetric sparse matrix (LD) + identity, then multiply with mean/variance.
    """
    n = data.x.shape[0]
    diag = torch.stack([torch.arange(n), torch.arange(n)], dim=0)
    ones = torch.ones(n, dtype=data.edge_attr.dtype)

    i3 = torch.cat([data.edge_index, torch.flip(data.edge_index, dims=[0]), diag], dim=1)
    v3 = torch.cat([data.edge_attr, data.edge_attr, ones])

    s3 = torch.sparse_coo_tensor(i3, v3, (n, n))
    v3_squared = v3 ** 2
    s3_var = torch.sparse_coo_tensor(i3, v3_squared, (n, n))

    # move to CPU for sparse mm if needed (depends on PyTorch build)
    prior_mean_cpu = prior_mean.cpu()
    prior_var_cpu = torch.exp(prior_logvar).cpu()

    all_mean = torch.sparse.mm(s3, prior_mean_cpu)
    all_var = torch.sparse.mm(s3_var, prior_var_cpu)
    return all_mean, all_var

# -----------------------------
# Train / Validate / Test
# -----------------------------
def adjust_lr_warmup(optimizer: torch.optim.Optimizer, epoch: int, warmup_epochs: int, base_lr: float) -> None:
    if epoch < warmup_epochs:
        lr = base_lr * float(epoch + 1) / warmup_epochs
        for g in optimizer.param_groups:
            g["lr"] = lr

def train_one_epoch(
    model: FCUncertainty,
    packs: Dict[int, ChrPack],
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    criterion,
) -> float:
    model.train()
    running = 0.0
    for chrom in np.random.permutation(list(packs.keys())):
        pack = packs[chrom]
        optimizer.zero_grad()

        mean, logvar = model(pack.annot, pack.freq_basis)  # on device
        out_mean, out_var = apply_sparse_ld(pack.data, mean, logvar)  # CPU sparse
        loss = criterion(out_mean.reshape(-1), out_var.reshape(-1), pack.target_y)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running += float(loss.item())
    return running

@torch.no_grad()
def evaluate(
    model: FCUncertainty,
    packs: Dict[int, ChrPack],
    criterion,
) -> float:
    model.eval()
    total = 0.0
    for chrom in packs:
        pack = packs[chrom]
        mean, logvar = model(pack.annot, pack.freq_basis)
        out_mean, out_var = apply_sparse_ld(pack.data, mean, logvar)
        val_loss = criterion(out_mean.reshape(-1), out_var.reshape(-1), pack.target_y)
        total += float(val_loss.item())
    return total

@torch.no_grad()
def export_priors(
    model: FCUncertainty,
    packs: Dict[int, ChrPack],
    summaries_df: pd.DataFrame,
    out_dir: Path,
    tag: str,
) -> None:
    model.eval()
    means_concat: List[torch.Tensor] = []
    logvars_concat: List[torch.Tensor] = []

    for chrom in range(1, 23):
        pack = packs[chrom]
        mean, logvar = model(pack.annot, pack.freq_basis)
        means_concat.append(mean.cpu())
        logvars_concat.append(logvar.cpu())

        torch.save(mean, out_dir / f"{tag}_prior_chr{chrom}.pt")
        torch.save(logvar, out_dir / f"{tag}_prior_logvar_chr{chrom}.pt")

    means_all = torch.cat(means_concat).numpy()
    logvars_all = torch.cat(logvars_concat).numpy()

    out = summaries_df[["Predictor"]].copy()
    out["Heritability"] = means_all
    out["logvar"] = logvars_all

    out.to_csv(out_dir / f"{tag}_ind_her_and_logvar.csv", index=False)

# -----------------------------
# Main
# -----------------------------
def main() -> None:
    setup_logger("INFO")
    cfg = parse_args()
    set_deterministic(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    ensure_dir(cfg.out_dir)
    out_dir = Path(cfg.out_dir)

    # load
    ref_maf = load_ref_maf(cfg.path_ref_maf)
    annot = load_annotation(cfg)
    summaries = load_summaries(cfg)

    logging.info("Preprocessing...")
    annot = add_chr_column(annot, "Predictor")
    summaries = add_chr_column(summaries, "Predictor")

    logging.info("Building chromosome packs...")
    packs = build_chr_packs(cfg, annot, summaries, ref_maf, device)

    # model / optim
    model = FCUncertainty(pca=cfg.pca).to(device)
    optimizer = Lamb(model.parameters(), lr=cfg.lr, weight_decay=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, threshold=1e-2, min_lr=1e-6)
    criterion = gaussian_nll
    early = EarlyStopping(patience=cfg.patience, delta=0.01)

    logging.info("Start training...")
    best_epoch = -1
    best_val = float("inf")

    for epoch in range(cfg.epochs):
        t0 = time.time()
        adjust_lr_warmup(optimizer, epoch, cfg.warmup_epochs, cfg.lr)

        train_loss = train_one_epoch(model, packs, device, optimizer, criterion)
        val_loss = evaluate(model, packs, criterion)
        scheduler.step(train_loss)

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), out_dir / f"model_best.pt")

        early(val_loss)
        elapsed = time.time() - t0
        mem = psutil.virtual_memory().percent
        logging.info(
            f"Epoch {epoch:03d} | train={train_loss:.4f} | val={val_loss:.4f} | "
            f"best@{best_epoch}({best_val:.4f}) | lr={optimizer.param_groups[0]['lr']:.2e} | {elapsed:.1f}s | mem={mem:.1f}%"
        )

        if early.early_stop:
            logging.info(f"Early stopping at epoch {epoch}. Best epoch = {best_epoch}.")
            break

    # reload best and export priors
    model.load_state_dict(torch.load(out_dir / "model_best.pt", map_location=device))
    tag = f"deepu_prs_{cfg.ver or 'v'}_r2-{cfg.r2_coverage}_lr{cfg.lr}_seed{cfg.seed}"
    export_priors(model, packs, summaries, out_dir, tag=tag)
    logging.info("Done.")

if __name__ == "__main__":
    main()
