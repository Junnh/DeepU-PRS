#!/bin/bash
# DeepU-PRS ensemble runner — trains M deep-ensemble members and aggregates them.
#
# Steps performed:
#   1. train.py     × M seeds     (deep-ensemble members)
#   2. make_ivw_prior.py           (IVW h^2 + simple-mean sigma^2)
#   3. make_adaptive_prior.py      (closed-form adaptive shrinkage)
# Downstream LDAK MegaPRS BayesR is invoked separately (see README).
#
# Usage:
#   ./scripts/run_ensemble.sh <data_root> <file_path> <biomarker>
# Example (HDL, 205-feature input):
#   ./scripts/run_ensemble.sh ./data HDL hdl

set -u

DATA_ROOT=${1:-./data}
FILE_PATH=${2:-HDL}
BIOMARKER=${3:-hdl}

# Feature set: 'enformer_new' = 205 (functional + Enformer PCA); 'deep_imp0' = 65 (functional only).
PCA_FOLDER='enformer_new'

# LD subdir (chrld_<c>.npy and chr<c>_edge_index.npy live under ${LD_ROOT}/${R2CUT}/).
LD_ROOT=./ld
R2CUT='cut_0.01'

# Annotation csv (205-feature).
ANNOT_205=${DATA_ROOT}/annot_205.csv

# plink .frq reference allele-frequency file.
MAF=${DATA_ROOT}/plink.frq

# LR + version tag.
LR='0.001'
VER='18607'

# Ensemble seeds. Match the paper.
SEED_ORDER='2023 2024 2025 2026 2027'

OUT_BASE=${DATA_ROOT}/${FILE_PATH}/${PCA_FOLDER}
mkdir -p "${OUT_BASE}"

echo "================================================================="
echo "DeepU-PRS ensemble runner"
echo "  trait      : ${FILE_PATH} (${BIOMARKER})"
echo "  data_root  : ${DATA_ROOT}"
echo "  feature set: ${PCA_FOLDER}"
echo "  seeds      : ${SEED_ORDER}"
echo "  version    : ${VER}"
echo "================================================================="

LOG_FILE=${OUT_BASE}/compute_log_${FILE_PATH}_v${VER}.txt
echo "=== NN log for ${FILE_PATH} (VER=${VER}) on $(hostname) $(date) ===" > "${LOG_FILE}"

# ---------- 1. Train M ensemble members ----------
for SEED in ${SEED_ORDER}; do
    echo "Starting seed ${SEED} at $(date)" | tee -a "${LOG_FILE}"
    START_TIME=$(date +%s)

    OUT_FILE=${OUT_BASE}/f${VER}_tissue_v37_sub2_seed${SEED}_r2-${R2CUT}_lr${LR}_train.out
    : > "${OUT_FILE}"

    python -u train.py \
        --data_root   "${DATA_ROOT}" \
        --file_path   "${FILE_PATH}" \
        --biomarker   "${BIOMARKER}" \
        --pca         True \
        --lr          ${LR} \
        --r2_coverage ${R2CUT} \
        --ld_root     "${LD_ROOT}" \
        --maf         "${MAF}" \
        --annot_205   "${ANNOT_205}" \
        --seed        ${SEED} \
        --ver         ${VER} \
        > "${OUT_FILE}"

    ELAPSED=$(( $(date +%s) - START_TIME ))
    echo "Seed ${SEED} NN done in ${ELAPSED} s." | tee -a "${LOG_FILE}"
    echo "---------------------------------------------------" >> "${LOG_FILE}"
done

# ---------- 2. IVW h^2 + simple-mean sigma^2 across seeds ----------
python make_ivw_prior.py \
    --input_dir   "${OUT_BASE}" \
    --seeds       ${SEED_ORDER} \
    --ver         ${VER} \
    --lr          ${LR} \
    --r2_coverage ${R2CUT} \
    --out_dir     "${OUT_BASE}"

# ---------- 3. Adaptive shrinkage prior ----------
python make_adaptive_prior.py \
    --h2_ivw  "${OUT_BASE}/h2_ivw.ind.her.pos.ens_ivw" \
    --sigma2  "${OUT_BASE}/sigma2_simple.ind.var.ens_simple" \
    --out_dir "${OUT_BASE}"

echo ""
echo "==============================================="
echo "DeepU-PRS pipeline complete on $(hostname) at $(date)"
echo "Adaptive prior: ${OUT_BASE}/_noamb.ind.her.pos.adaptive"
echo "Next: feed this file to LDAK --mega-prs via --ind-hers (see README)."
echo "==============================================="
