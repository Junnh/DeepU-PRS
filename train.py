"""DeepU-PRS training — single ensemble member (production v7).

Trains one deep-ensemble member that maps per-SNP functional annotations to
(heritability prior h^2, log aleatoric variance).  Run with M different seeds
to build the ensemble consumed downstream by make_ivw_prior.py and
make_adaptive_prior.py.

Inputs (paths configurable via CLI):
  --data_root    base dir containing <file_path>/neale.train.summaries and per-chr caches
  --file_path    trait subdir under data_root (e.g. HDL)
  --biomarker    lowercase label used in log messages
  --pca True     205-feature input (functional + Enformer PCA);  False -> 65 features
  --lr           learning rate
  --r2_coverage  LD subdir name (default cut_0.01)
  --seed         RNG seed
  --ver          version tag used in output filenames
"""

import argparse
import gc
import os
import time

import numpy as np
import pandas as pd
import psutil
import torch
from torch import nn
from torch_geometric.data import Data
from pytorch_lamb import Lamb


parser = argparse.ArgumentParser()
parser.add_argument('--data_root',   type=str, required=True,
                    help='base dir containing <file_path>/neale.train.summaries and tensor caches')
parser.add_argument('--file_path',   type=str, required=True, help='trait subdir (e.g. HDL)')
parser.add_argument('--biomarker',   type=str, required=True, help='lowercase label')
parser.add_argument('--pca',         type=lambda s: str(s).lower() == 'true', default=False,
                    help='True -> 205-feature input; False -> 65-feature')
parser.add_argument('--lr',          type=float, required=True)
parser.add_argument('--r2_coverage', type=str, default='cut_0.01')
parser.add_argument('--ld_root',     type=str, required=True,
                    help='directory containing <r2_coverage>/chrld_<c>.npy and chr<c>_edge_index.npy')
parser.add_argument('--maf',         type=str, required=True,
                    help='plink .frq reference allele-frequency file')
parser.add_argument('--annot_65',    type=str, default=None,
                    help='65-feature annotation csv (used when --pca False)')
parser.add_argument('--annot_205',   type=str, default=None,
                    help='205-feature annotation csv (used when --pca True)')
parser.add_argument('--seed',        type=str, required=True)
parser.add_argument('--ver',         type=str, required=True, help='tag used in output filenames')
args = parser.parse_args()

print(args.file_path, args.biomarker, args.pca, args.r2_coverage, args.seed, args.ver)

# Seed all RNGs so that the same --seed reproduces the released pretrained weights.
_seed_int = int(args.seed)
np.random.seed(_seed_int)
torch.manual_seed(_seed_int)
torch.cuda.manual_seed_all(_seed_int)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

folder = '/enformer_new' if args.pca else '/deep_imp0'

DATA_ROOT = args.data_root.rstrip('/')


def read_table(path, **kwargs):
    pq_path = path + '.parquet'
    if os.path.exists(pq_path):
        return pd.read_parquet(pq_path)
    df = pd.read_csv(path, **kwargs)
    try:
        df.to_parquet(pq_path)
        print(f'parquet sidecar written: {pq_path}')
    except Exception as exc:
        print(f'parquet sidecar write skipped ({exc})')
    return df


chr_list = np.arange(1, 23)

cache_dir = os.path.join(
    DATA_ROOT,
    'tensor_cache_' + args.r2_coverage.replace('/', '_') + '_' + str(args.pca) + '_False'
)
sparse_cache_dir = cache_dir + '_sparse'
os.makedirs(cache_dir, exist_ok=True)
os.makedirs(sparse_cache_dir, exist_ok=True)


def _chr_cache_ok(c):
    small = os.path.join(cache_dir, f'chr_{c}_small.pt')
    csr   = os.path.join(sparse_cache_dir, f'chr_{c}_csr.pt')
    return os.path.exists(small) and os.path.exists(csr)


all_cached = all(_chr_cache_ok(c) for c in chr_list)

summaries = read_table(os.path.join(DATA_ROOT, args.file_path, 'neale.train.summaries'),
                       delim_whitespace=True)

# Mask BEFORE the numerical floor so we know which SNPs were originally invalid.
summaries['valid_mask'] = summaries['Stat'] > 0
n_invalid = int((~summaries['valid_mask']).sum())
print(f'zero/negative-stat SNPs masked from loss: {n_invalid} / {len(summaries)} '
      f'({100.0 * n_invalid / len(summaries):.3f}%)')
summaries.loc[summaries['Stat'] <= 0, 'Stat'] = 1e-06  # numerical floor; masked anyway
summaries['chr'] = summaries['Predictor'].apply(lambda s: s.split(':')[0])
print('summaries: ', summaries.shape)

data_dict = {}

if all_cached:
    print('All processed tensors found in cache. Skipping raw data preprocessing.')
else:
    print('Cache missing or incomplete. Starting full data preprocessing.')

    ref_maf = read_table(args.maf, delim_whitespace=True)

    if args.pca:
        assert args.annot_205 is not None, '--annot_205 required when --pca True'
        annot = read_table(args.annot_205, engine='c')
    else:
        assert args.annot_65 is not None, '--annot_65 required when --pca False'
        annot = read_table(args.annot_65, engine='c')

    print('usecols ', annot.shape[1])
    annot['chr'] = annot['Predictor'].apply(lambda s: s.split(':')[0])

    annot_grouped     = annot.groupby('chr')
    summaries_grouped = summaries.groupby('chr')
    ref_maf_grouped   = ref_maf.groupby('CHR')

    print('Start mapping df')

    for chr_num in chr_list:
        str_chr = str(chr_num)
        annot_chr     = annot_grouped.get_group(str_chr)
        summaries_chr = summaries_grouped.get_group(str_chr)
        ref_maf_chr   = ref_maf_grouped.get_group(int(chr_num))

        valid_snps = summaries_chr['Predictor'].unique()
        ref_maf_filt_chr = ref_maf_chr[ref_maf_chr['SNP'].isin(valid_snps)]

        chrld = np.load(
            os.path.join(args.ld_root, args.r2_coverage, f'chrld_{chr_num}.npy'),
            allow_pickle=True,
        )

        valid_snps_arr = summaries_chr['Predictor'].values
        search = (pd.Series(chrld[:, 0]).isin(valid_snps_arr) &
                  pd.Series(chrld[:, 1]).isin(valid_snps_arr)).values
        edge_attr = torch.tensor(chrld[search][:, 2].astype(np.float32), dtype=torch.float32)

        edge_index = torch.tensor(
            np.load(os.path.join(args.ld_root, args.r2_coverage, f'chr{chr_num}_edge_index.npy')).transpose(),
            dtype=torch.long,
        )

        # target = log(Stat).  1e-06 floor applied above, so log is finite.
        stat_arr = summaries_chr['Stat'].values.astype(np.float32)
        n_arr    = summaries_chr['n'].values.astype(np.float32)
        mask_arr = summaries_chr['valid_mask'].values.astype(bool)

        Y      = torch.tensor(np.log(stat_arr), dtype=torch.float32)
        n_t    = torch.tensor(n_arr, dtype=torch.float32)
        mask_t = torch.tensor(mask_arr, dtype=torch.bool)

        annot_vals = annot_chr.drop(['Predictor', 'chr'], axis=1)
        annot_df   = torch.tensor(annot_vals.values, dtype=torch.float32)

        maf      = torch.tensor(ref_maf_filt_chr['MAF'].values, dtype=torch.float32)
        maf_bias = ((maf * (1 - maf)) ** 0.75).reshape(-1, 1)

        num_nodes = len(annot_chr['Predictor'].unique())
        data = Data(num_nodes=num_nodes, edge_index=edge_index, edge_attr=edge_attr,
                    y=Y, n=n_t, mask=mask_t)

        data_dict[chr_num] = {'data': data, 'annot_df': annot_df, 'maf_bias': maf_bias}

        del chrld, search, annot_vals, ref_maf_filt_chr, annot_chr, summaries_chr, ref_maf_chr
        gc.collect()
        print(psutil.virtual_memory().percent)

    del annot, ref_maf, annot_grouped, summaries_grouped, ref_maf_grouped
    gc.collect()

print('Data preprocessing finished')


class FC(nn.Module):
    def __init__(self):
        super().__init__()
        if args.pca:
            self.fc1 = nn.Linear(205, 128); self.bn1 = nn.BatchNorm1d(128)
            self.fc3 = nn.Linear(128,  96); self.bn3 = nn.BatchNorm1d(96)
            self.fc4 = nn.Linear(96,   64); self.bn4 = nn.BatchNorm1d(64)

            self.avg_fc = nn.Linear(64, 32); self.avg_bn = nn.BatchNorm1d(32)
            self.var_fc = nn.Linear(64, 32); self.var_bn = nn.BatchNorm1d(32)
            self.avg = nn.Linear(32, 1)
            self.var = nn.Linear(32, 1)
        else:
            self.fc1 = nn.Linear(65, 32)
            self.fc2 = nn.Linear(32, 10)

        self.softplus   = nn.Softplus()
        self.activation = nn.SiLU()
        self.activations = {}
        self.activation.register_forward_hook(self.get_activation('activation'))
        # log-Stat target lives at ~1e-6 scale; scaling factor matches.
        self.scaling_factor = nn.Parameter(torch.tensor(1e-6))

    def get_activation(self, name):
        def hook(_m, _in, output):
            self.activations[name] = (output.mean().detach(), output.std().detach())
        return hook

    def forward(self, x, freq_basis, task_num):
        x = self.activation(self.bn1(self.fc1(x)))
        x = self.activation(self.bn3(self.fc3(x)))
        x = self.activation(self.bn4(self.fc4(x)))

        mean   = self.activation(self.avg_bn(self.avg_fc(x)))
        logvar = self.activation(self.var_bn(self.var_fc(x)))
        mean   = freq_basis * self.softplus(self.avg(mean)) * self.scaling_factor
        logvar = self.var(logvar)

        print('Chr running ', task_num)
        mean   = mean.reshape(-1, 1)
        logvar = logvar.reshape(-1, 1)
        print('Scaling Factor: ', self.scaling_factor)
        print('Total sum: ',  torch.sum(mean))
        print('Pos sum: ',    torch.sum(mean[mean > 0]))
        print('Variance max min: ', torch.max(torch.exp(logvar)), torch.min(torch.exp(logvar)))
        print('Var mean: ',   torch.mean(torch.exp(logvar)))
        return mean, logvar


class Pass(nn.Module):
    def __init__(self, FC_net, dataset):
        super().__init__()
        self.FC_net = FC_net
        self.dataset = dataset

    def forward(self, chr_num, return_prior=False):
        entry = self.dataset[chr_num]
        data    = entry['data']
        annot   = entry['annot'].to(device)
        freq_bs = entry['freq_bs'].to(device)
        size    = entry['size']

        y         = data.y.to(device)
        n_per_snp = data.n.to(device)
        mask      = data.mask.to(device)

        prior, logvar = self.FC_net(annot, freq_bs, chr_num)
        print('prior calculated..')

        crow_cpu, col_cpu, vals_cpu, _size = torch.load(entry['csr_path'], weights_only=False)
        crow = crow_cpu.to(device, non_blocking=True)
        col  = col_cpu .to(device, non_blocking=True)
        vals = vals_cpu.to(device, non_blocking=True)
        del crow_cpu, col_cpu, vals_cpu
        print('full sparse bf', psutil.virtual_memory().percent, 'memory')

        vals_sq = (vals * vals).contiguous()

        s3      = torch.sparse_csr_tensor(crow, col, vals,    (size, size))
        all_snp = torch.sparse.mm(s3, prior)
        del s3
        torch.cuda.synchronize()

        s3_var      = torch.sparse_csr_tensor(crow, col, vals_sq, (size, size))
        all_snp_var = torch.sparse.mm(s3_var, torch.exp(logvar))
        del s3_var, crow, col, vals, vals_sq

        print('full sparse af ', psutil.virtual_memory().percent, 'memory')

        if return_prior:
            return all_snp, y, all_snp_var, prior, logvar, n_per_snp, mask
        return all_snp, y, all_snp_var, n_per_snp, mask


def adjust_learning_rate(optimizer, epoch, warmup_epochs, base_lr):
    lr = base_lr * float(epoch + 1) / warmup_epochs if epoch < warmup_epochs else base_lr
    for pg in optimizer.param_groups:
        pg['lr'] = lr


FC_net = FC()


def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


FC_net.apply(initialize_weights)


# ---------------- Build sparse (CSR) cache and Dataset dict ------------------
def _build_csr_from_i3_v3(i3, v3, size):
    s_coo = torch.sparse_coo_tensor(i3, v3, (size, size)).coalesce()
    s_csr = s_coo.to_sparse_csr()
    crow = s_csr.crow_indices().to(torch.int32).contiguous()
    col  = s_csr.col_indices ().to(torch.int32).contiguous()
    vals = s_csr.values().contiguous()
    del s_coo, s_csr
    return crow, col, vals


dataset = {}
for chr_num in chr_list:
    small_path = os.path.join(cache_dir,        f'chr_{chr_num}_small.pt')
    csr_path   = os.path.join(sparse_cache_dir, f'chr_{chr_num}_csr.pt')

    if os.path.exists(small_path):
        d_t, annot_df, maf_bias = torch.load(small_path, weights_only=False)
    else:
        d_dict = data_dict[chr_num]
        d_t      = d_dict['data']
        annot_df = d_dict['annot_df']
        maf_bias = d_dict['maf_bias']
        torch.save([d_t, annot_df, maf_bias], small_path)

    size = d_t.num_nodes

    if not os.path.exists(csr_path):
        d = data_dict[chr_num]['data'] if chr_num in data_dict else d_t
        N = d.num_nodes
        diag = torch.arange(N).reshape(1, -1).expand(2, -1)
        one  = torch.ones(N, dtype=torch.float32)
        i3   = torch.cat([d.edge_index, torch.flip(d.edge_index, dims=[0]), diag], axis=1).contiguous()
        v3   = torch.cat([d.edge_attr, d.edge_attr, one])
        del diag, one
        if chr_num in data_dict:
            del data_dict[chr_num]

        crow, col, vals = _build_csr_from_i3_v3(i3, v3, size)
        torch.save([crow, col, vals, size], csr_path)
        print(f'chr {chr_num}: CSR cache written (size={size}, nnz={vals.numel()})  '
              f'{psutil.virtual_memory().percent}% mem')
        del i3, v3, crow, col, vals

    gc.collect()

    dataset[chr_num] = {
        'data':     d_t,
        'annot':    annot_df,
        'freq_bs':  maf_bias,
        'csr_path': csr_path,
        'size':     size,
    }


model         = Pass(FC_net, dataset)
epochs        = 40
patience      = 3
warmup_epochs = 10
base_lr       = args.lr


class RelativeEarlyStopping:
    """Stop when the metric fails to improve by >= min_rel_improve for `patience` calls."""
    def __init__(self, patience=3, min_rel_improve=0.01, verbose=True):
        self.patience = patience
        self.min_rel_improve = min_rel_improve
        self.verbose = verbose
        self.best = float('inf')
        self.best_epoch = -1
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss, model=None, epoch=None):
        if self.best == float('inf'):
            self.best = val_loss
            self.best_epoch = epoch
            self.counter = 0
            if self.verbose:
                print(f'Validation first: {val_loss:.4e}')
            return
        threshold = self.best - abs(self.best) * self.min_rel_improve
        if val_loss < threshold:
            if self.verbose:
                print(f'Validation improved ({self.best:.4e} -> {val_loss:.4e})')
            self.best = val_loss
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f'No >{self.min_rel_improve * 100:.1f}% improvement '
                      f'({self.counter}/{self.patience}): cur {val_loss:.4e} vs best {self.best:.4e}')
            if self.counter >= self.patience:
                self.early_stop = True


early_stopping = RelativeEarlyStopping(patience=patience, min_rel_improve=0.01)

optimizer = Lamb(model.parameters(), lr=args.lr, weight_decay=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3, threshold=1e-2, min_lr=1e-6)


def nll_gaussian(y_mean, y_var, y_true):
    y_var = torch.clamp(y_var, min=1e-8, max=1e2)
    nll = 0.5 * torch.log(y_var) + 0.5 * ((y_mean - y_true) ** 2) / y_var
    print('MSE loss: ',     torch.mean(0.5 * ((y_mean - y_true) ** 2) / y_var))
    print('variance loss: ', torch.mean(0.5 * torch.log(y_var)))
    return torch.mean(nll)


def compute_loss(output, allvar, label, n_per_snp, mask):
    """Match in log-Stat space: prediction = log(1 + n * (R^2 · h^2)); target = log(Stat)."""
    out_flat = output.reshape(-1)
    var_flat = allvar.reshape(-1)
    pred_log = torch.log1p(n_per_snp * torch.clamp(out_flat, min=0.0))
    loss = nll_gaussian(pred_log[mask], var_flat[mask], label[mask])
    with torch.no_grad():
        raw_mse = torch.mean((pred_log[mask] - label[mask]) ** 2).item()
    return loss, raw_mse


FC_net.to(device)
print('Start training..')

loss_tracking = []
val_loss_tracking = []
mse_tracking = []
val_mse_tracking = []
chr_train = np.arange(1, 23)

e = 0
for e in range(epochs):
    start = time.time()
    running_loss = 0
    running_mse  = 0

    epoch_grad_norm = 0
    epoch_act_means = []
    epoch_act_stds  = []
    np.random.shuffle(chr_train)
    model.train()
    if e < warmup_epochs:
        adjust_learning_rate(optimizer, e, warmup_epochs, base_lr)

    for chr_num in chr_train:
        optimizer.zero_grad()
        output, label, allvar, n_per_snp, mask = model(chr_num)

        if e == 0:
            assert not torch.isnan(output).any(), 'Outputs contain NaN'
            assert not torch.isinf(output).any(), 'Outputs contain Inf'
            assert not torch.isnan(allvar).any(), 'Variance contains NaN'
            assert not torch.isinf(allvar).any(), 'Variance contains Inf'

        loss, raw_mse = compute_loss(output, allvar, label, n_per_snp, mask)
        running_mse += raw_mse
        loss.backward()

        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        print('chr norm: ', total_norm)
        epoch_grad_norm += total_norm

        act_stats = model.FC_net.activations.get('activation', None)
        if act_stats is not None:
            epoch_act_means.append(act_stats[0])
            epoch_act_stds .append(act_stats[1])

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        optimizer.step()

        running_loss += loss.item()
        print(f'Chr loss: {loss.item()}')

    avg_grad_norm = epoch_grad_norm / len(chr_train)
    if epoch_act_means:
        avg_act_mean = torch.stack(epoch_act_means).mean().item()
        avg_act_std  = torch.stack(epoch_act_stds ).mean().item()
    else:
        avg_act_mean = 0
        avg_act_std  = 0
    print(f'Epoch {e} Avg Gradient Norm: {avg_grad_norm}')
    print(f'Epoch {e} Avg Activation Mean: {avg_act_mean}, Std: {avg_act_std}')

    print(gc.collect())
    print(f'Total Training loss: {running_loss}')
    print(f'Total Training MSE: {running_mse}')
    loss_tracking.append(running_loss)
    mse_tracking .append(running_mse)
    print(time.time() - start)
    print('epoch', e)

    if e > warmup_epochs:
        with torch.no_grad():
            model.eval()
            val_running_loss = 0
            val_running_mse  = 0
            start = time.time()
            for chr_num in chr_train:
                val_output, val_label, val_allvar, val_n, val_mask = model(chr_num)
                val_loss, val_raw_mse = compute_loss(val_output, val_allvar, val_label, val_n, val_mask)
                val_running_loss += val_loss.item()
                val_running_mse  += val_raw_mse

            print('Valid loss: ', val_running_loss)
            print('Valid MSE: ',  val_running_mse)
            print(time.time() - start)
            val_loss_tracking.append(val_running_loss)
            val_mse_tracking .append(val_running_mse)

            early_stopping(val_running_loss, model)

            ckpt_path = os.path.join(
                DATA_ROOT, args.file_path, folder.lstrip('/'),
                f'f{args.ver}_tissue_v37_sub2_param_e{e}_lr{args.lr}_r2-{args.r2_coverage}'
                f'_imp3.{args.seed}.pt',
            )
            torch.save(model.state_dict(), ckpt_path)

            if early_stopping.early_stop:
                print('Early stopping best epoch =', e - patience)
                break

    scheduler.step(running_loss)
    print('lr: ', optimizer.param_groups[0]['lr'])


# ---------------- Inference from best checkpoint ------------------
model = Pass(FC_net, dataset)
param = os.path.join(
    DATA_ROOT, args.file_path, folder.lstrip('/'),
    f'f{args.ver}_tissue_v37_sub2_param_e{e - patience}_lr{args.lr}_r2-{args.r2_coverage}'
    f'_imp3.{args.seed}.pt',
)
model.load_state_dict(torch.load(param, weights_only=False))
model.to(device)

temp_list     = []
temp_sig_list = []
with torch.no_grad():
    model.eval()
    test_running_loss = 0
    for chr_num in chr_list:
        output, label, allvar, pr, logvar, n_per_snp, mask = model(chr_num, return_prior=True)
        test_loss, _ = compute_loss(output, allvar, label, n_per_snp, mask)
        test_running_loss += test_loss.item()

        pref = os.path.join(
            DATA_ROOT, args.file_path, folder.lstrip('/'),
            f'f{args.ver}_tissue_v37_sub2_imp2_prior_e{e - patience}_chr_{chr_num}'
            f'_r2-{args.r2_coverage}_lr{args.lr}_seed{args.seed}',
        )
        torch.save(pr,     pref)
        torch.save(logvar, pref.replace('_prior_', '_prior_logvar_'))

        temp_list    .append(pr    .detach().cpu())
        temp_sig_list.append(logvar.detach().cpu())
        del pr, logvar
        print('chr ', chr_num)

    temp     = torch.cat(temp_list,     dim=0)
    temp_sig = torch.cat(temp_sig_list, dim=0)
    print(f'Test loss: {test_running_loss}')

print('End first step..')
print(torch.sum(temp))
print(torch.sum(temp[temp > 0]))

summaries['Heritability'] = temp.numpy()
deep_her = summaries[['Predictor', 'Heritability']]

out_pref = os.path.join(
    DATA_ROOT, args.file_path, folder.lstrip('/'),
    f'f{args.ver}_tissue_v37_sub2_r2-{args.r2_coverage}_lr{args.lr}_noamb',
)

deep_her.to_csv(f'{out_pref}.ind.her.{args.seed}',                    sep='\t', index=None, header=None)
deep_her[deep_her['Heritability'] > 0].to_csv(
    f'{out_pref}.ind.her.pos.{args.seed}',                            sep='\t', index=None, header=None)
summaries[summaries['Predictor'].isin(
    deep_her[deep_her['Heritability'] > 0]['Predictor'])]['Predictor'].to_csv(
    f'{out_pref}_snps.{args.seed}.txt',                               sep='\t', index=None, header=None)

deep_her['logvar'] = temp_sig.numpy()
deep_her.to_csv(f'{out_pref}.ind.her.logvar{args.seed}.csv',          index=None)

print('Prior Done')
