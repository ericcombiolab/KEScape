import pandas as pd
import os
import socket
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.optim.lr_scheduler import LambdaLR
from scipy.stats import spearmanr
from model import FitnessPredictor, BACKBONE_CKPT_PATH
from data import PooledFitnessDataset, collate_variable_length
from datetime import timedelta
import argparse
from contextlib import nullcontext

class CSVHuberLoss(nn.Module):
    def __init__(self, delta=1.0, weight_mean=1.0):
        super(CSVHuberLoss, self).__init__()
        self.delta = delta
        self.weight_mean = float(weight_mean) if weight_mean > 0 else 1.0

    def forward(self, preds, targets, weights):
        targets = targets.float()
        preds = preds.float()
        huber = F.huber_loss(preds, targets, delta=self.delta, reduction='none')
        w = weights.float() / self.weight_mean
        return (w * huber).mean()

def setup():
    dist.init_process_group(backend='nccl', init_method='env://', timeout=timedelta(hours=3))

def cleanup():
    dist.destroy_process_group()

def load_data(rank, world_size, data_path, batch_size, is_train, label_col, dist_label_col,
              winsor=(-1.5, 2.5), nprocs=10):
    df = pd.read_csv('{}'.format(data_path))
    raw_seqs = df.loc[:, 'wildtype_sequence'].values
    mut_seqs = df.loc[:, 'mutant_sequence'].values
    if is_train:
        judge_same = (raw_seqs == mut_seqs)
        if sum(judge_same) > 0:
            print('There are {} samples with raw_seq == mut_seq'.format(sum(judge_same)))
            raise Exception('There are samples with raw_seq == mut_seq')
    labels = np.clip(df.loc[:, label_col].values.astype(np.float64),
                     winsor[0], winsor[1])
    weights = df.loc[:, 'sample_weight'].values
    dist_labels = np.clip(df.loc[:, dist_label_col].values.astype(np.float64),
                         winsor[0], winsor[1])
    dataset = PooledFitnessDataset(raw_seqs, mut_seqs, labels, weights, dist_labels,
                                   nprocs=nprocs)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=is_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=sampler,
                        collate_fn=collate_variable_length, pin_memory=True)
    return sampler, loader, len(df)

def load_validation_data(data_path, batch_size, label_col, dist_label_col,
                         winsor=(-1.5, 2.5), rank=0, world_size=1, nprocs=10):
    df = pd.read_csv('{}'.format(data_path))
    dataset = PooledFitnessDataset(
        df.loc[:, 'wildtype_sequence'].values,
        df.loc[:, 'mutant_sequence'].values,
        np.clip(df.loc[:, label_col].values.astype(np.float64),
                winsor[0], winsor[1]),
        df.loc[:, 'sample_weight'].values,
        np.clip(df.loc[:, dist_label_col].values.astype(np.float64),
                winsor[0], winsor[1]),
        nprocs=nprocs,
    )
    sampler = (DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                  shuffle=False, drop_last=False)
               if world_size > 1 else None)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=sampler,
                        collate_fn=collate_variable_length, pin_memory=True)
    return loader, len(df)

def compute_total_loss(
    immune_escape_score,
    dist_score,
    labels,
    weights,
    dist_labels,
    avg_l2_per_sample,
    criterion,
    cls_threshold=1.0,
):
    labels = labels.float()
    loss_immune_escape = criterion(immune_escape_score, labels, weights)
    loss_dist = criterion(dist_score, dist_labels.float(), weights)
    loss = loss_immune_escape + loss_dist

    pos_mask = labels >= cls_threshold
    neg_mask = ~pos_mask
    batch_stats = {
        'loss_dist': loss_dist.item(),
        'loss_immune_escape': loss_immune_escape.item(),
        'avg_l2_pos_sum': avg_l2_per_sample[pos_mask].sum().item() if pos_mask.any() else 0.0,
        'avg_l2_neg_sum': avg_l2_per_sample[neg_mask].sum().item() if neg_mask.any() else 0.0,
        'num_pos': int(pos_mask.sum().item()),
        'num_neg': int(neg_mask.sum().item()),
        'num_samples': int(labels.shape[0]),
    }
    return loss, batch_stats

def evaluate_model(
    model,
    val_loader,
    criterion,
    device,
    cls_threshold=1.0,
    rank=0,
    world_size=1,
    n_dataset=None,
):
    model.eval()
    if n_dataset is None:
        n_dataset = len(val_loader.dataset)
    n_real = (n_dataset - rank + world_size - 1) // world_size if rank < n_dataset else 0
    remaining = n_real
    val_preds = []
    val_labels = []
    sums = np.zeros(3, dtype=np.float64)

    with torch.no_grad():
        for (batch_raw_seqs, batch_mut_seqs, batch_mask_raw, batch_mask_mut,
             batch_labels, batch_weights, batch_dist_labels,
             batch_raw_mutations, batch_mut_mutations) in val_loader:
            if remaining <= 0:
                break
            k = min(remaining, batch_labels.shape[0])
            remaining -= k
            batch_raw_seqs = batch_raw_seqs[:k].to(device)
            batch_mut_seqs = batch_mut_seqs[:k].to(device)
            batch_mask_raw = batch_mask_raw[:k].to(device)
            batch_mask_mut = batch_mask_mut[:k].to(device)
            batch_labels = batch_labels[:k].to(device)
            batch_weights = batch_weights[:k].to(device)
            batch_dist_labels = batch_dist_labels[:k].to(device)
            batch_raw_mutations = batch_raw_mutations[:k].to(device)
            batch_mut_mutations = batch_mut_mutations[:k].to(device)

            immune_escape_score, avg_l2_per_sample, dist_score = model(
                batch_raw_seqs, batch_mut_seqs, batch_mask_raw, batch_mask_mut,
                batch_raw_mutations, batch_mut_mutations)
            loss, batch_stats = compute_total_loss(
                immune_escape_score, dist_score, batch_labels, batch_weights,
                batch_dist_labels, avg_l2_per_sample, criterion,
                cls_threshold,
            )
            n_b = batch_stats['num_samples']
            sums[0] += loss.item() * n_b
            sums[1] += batch_stats['loss_dist'] * n_b
            sums[2] += batch_stats['loss_immune_escape'] * n_b

            val_preds.append(immune_escape_score.detach().float().cpu().numpy())
            val_labels.append(batch_labels.float().cpu().numpy())

    val_preds = np.concatenate(val_preds) if val_preds else np.zeros(0, dtype=np.float64)
    val_labels = np.concatenate(val_labels) if val_labels else np.zeros(0, dtype=np.float64)
    if world_size > 1:
        gathered = [None] * world_size
        dist.all_gather_object(
            gathered, (val_preds.astype(np.float64), val_labels.astype(np.float64), sums))
        val_preds = np.concatenate([g[0] for g in gathered])
        val_labels = np.concatenate([g[1] for g in gathered])
        sums = np.sum(np.stack([g[2] for g in gathered]), axis=0)

    n = val_labels.shape[0]
    metrics = {
        'loss': sums[0] / n,
        'loss_dist': sums[1] / n,
        'loss_immune_escape': sums[2] / n,
        'spearman_correlation': spearmanr(val_labels, val_preds)[0],
    }

    model.train()
    return metrics

def evaluate_all_datasets(model, val_loaders, criterion, device,
                          cls_threshold, rank, world_size):
    rows = []
    sp_vals = []
    for ds, loader, n_ds in val_loaders:
        m = evaluate_model(model, loader, criterion, device,
                           cls_threshold,
                           rank=rank, world_size=world_size, n_dataset=n_ds)
        m['dataset'] = ds
        rows.append(m)
        sp_vals.append(m['spearman_correlation'])
        if rank == 0:
            print('  valid %-14s Spearman %.4f (loss_immune_escape %.4f, loss_dist %.4f)'
                  % (ds, m['spearman_correlation'], m['loss_immune_escape'],
                     m['loss_dist']), flush=True)
    if rank == 0:
        print('  valid MEAN Spearman over %d datasets: %.4f'
              % (len(sp_vals), float(np.nanmean(sp_vals))), flush=True)
    return rows

def train_model(
    num_layers,
    num_heads,
    block_embed_dim,
    batch_size,
    eval_batchsize,
    data_path,
    protein,
    huber_delta,
    grad_clip,
    cls_threshold,
    label_col,
    dist_label_col,
    valid_datasets,
    std_dir,
    num_epochs,
    valid_split,
    dist_slope_frac,
    seed,
    winsor,
):
    setup()

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    train_sampler, train_loader, num_samples = load_data(
        rank, world_size, data_path, batch_size, is_train=True,
        label_col=label_col, dist_label_col=dist_label_col, winsor=winsor)

    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)

    model = FitnessPredictor(num_layers,num_heads,block_embed_dim).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    _calib_n = 256
    _calib_idx = torch.randperm(len(train_loader.dataset),
                                generator=torch.Generator().manual_seed(0))[:_calib_n].tolist()
    _calib_loader = DataLoader(train_loader.dataset, batch_size=eval_batchsize,
                               sampler=SubsetRandomSampler(_calib_idx),
                               collate_fn=collate_variable_length, pin_memory=True)
    model.eval()
    _l2s = []
    with torch.no_grad():
        for (_raw, _mut, _mraw, _mmut, _lbl, _w, _albl, _rmut, _mmutd) in _calib_loader:
            _out = model.module(
                _raw.to(device), _mut.to(device), _mraw.to(device), _mmut.to(device),
                _rmut.to(device), _mmutd.to(device))
            _l2s.append(_out[1].detach().float().cpu())
    _l2 = torch.cat(_l2s)
    _labels_t = torch.as_tensor(np.asarray(train_loader.dataset.dist_labels), dtype=torch.float32)
    model.module.calibrate_dist_head(
        _l2.mean(), _l2.std(), _labels_t.mean(), _labels_t.std(), dist_slope_frac)
    if rank == 0:
        print('dist_head calibrated: softplus(w)=%.4f, b=%.4f '
              '(l2 mean %.3f std %.3f; label mean %.3f std %.3f)'
              % (float(F.softplus(model.module.dist_head.weight)),
                 float(model.module.dist_head.bias),
                 float(_l2.mean()), float(_l2.std()),
                 float(_labels_t.mean()), float(_labels_t.std())), flush=True)
    model.train()

    eval_dir = '/home/datasets/cschwang/escape/evaluation/escape/DMS_subset_evescape_continuous/processed_data'
    val_loaders = []
    for ds in valid_datasets:
        path = os.path.join(eval_dir if valid_split == 'test' else std_dir,
                            '{}_{}_independent.csv'.format(valid_split, ds))
        vloader, n_ds = load_validation_data(path, eval_batchsize, label_col,
                                             dist_label_col, winsor,
                                             rank=rank, world_size=world_size)
        val_loaders.append((ds, vloader, n_ds))

    lr_params = 1e-3
    param_groups = [
        {"params": [p for p in model.parameters()], "lr": lr_params}
    ]

    weight_mean = float(np.mean(train_loader.dataset.weights))
    criterion = CSVHuberLoss(delta=huber_delta,
                                 weight_mean=weight_mean).to(device)
    if rank == 0:
        df_head = pd.read_csv(data_path)
        print('Training rows: %d; immune escape label col: %s (winsorized to %s); distance label col: %s (winsorized to %s); datasets: %s'
              % (len(df_head), label_col, tuple(winsor), dist_label_col, tuple(winsor),
                 ','.join(sorted(df_head['dataset'].unique()))), flush=True)
        print('sample_weight: min %.3g mean %.3g max %.3g'
              % (df_head['sample_weight'].min(), df_head['sample_weight'].mean(),
                 df_head['sample_weight'].max()), flush=True)
        _raw_dist = df_head[dist_label_col].values.astype(np.float64)
        print('distance label: %d rows (%.2f%%) winsorized to [%g, %g]'
              % (int((( _raw_dist < winsor[0]) | (_raw_dist > winsor[1])).sum()),
                 100.0 * float(((_raw_dist < winsor[0]) | (_raw_dist > winsor[1])).mean()),
                 winsor[0], winsor[1]), flush=True)
        _raw_lbl = df_head[label_col].values.astype(np.float64)
        print('main label: %d rows (%.2f%%) winsorized to [%g, %g]'
              % (int(((_raw_lbl < winsor[0]) | (_raw_lbl > winsor[1])).sum()),
                 100.0 * float(((_raw_lbl < winsor[0]) | (_raw_lbl > winsor[1])).mean()),
                 winsor[0], winsor[1]), flush=True)
        pos_fraction = float((np.clip(_raw_lbl, winsor[0], winsor[1]) >= cls_threshold).mean())
        print('Fraction of labels >= %s: %.4f' % (cls_threshold, pos_fraction), flush=True)
    validation_history = []
    result_path = f'./results/valid_performance_{protein}_pooled_seed{seed}.csv'
    if rank == 0:
        os.makedirs('./results', exist_ok=True)
    optimizer = optim.AdamW(param_groups, betas= (0.9,0.98), weight_decay=0.01)


    start_epoch = 0
    end_epoch = num_epochs
    num_epochs = end_epoch - start_epoch
    accumulation_batch_size = 150
    accumulation_steps = accumulation_batch_size // (batch_size*world_size)

    batches_per_epoch = int(np.ceil(num_samples/world_size/batch_size))
    steps_per_epoch = batches_per_epoch // accumulation_steps
    if batches_per_epoch % accumulation_steps != 0:
        steps_per_epoch += 1
    num_training_steps = steps_per_epoch * num_epochs

    num_warmup_steps = min(int(0.05*num_training_steps),500)
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.1, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    scheduler = LambdaLR(optimizer, lr_lambda)

    best_val_loss = float('inf')
    best_val_spearman = float('-inf')
    def _ckpt_name(tag):
        return ('./checkpoints/model_small_{}_{}_pooled_seed{}.pt'.format(tag, protein, seed))
    ckpt_meta = {
        'seed': seed,
        'dist_slope_frac': dist_slope_frac,
        'huber_delta': huber_delta,
        'grad_clip': grad_clip,
        'weight_mean': weight_mean,
        'label_col': label_col,
        'dist_label_col': dist_label_col,
        'winsor': list(winsor),
        'cls_threshold': cls_threshold,
        'num_layers': num_layers,
        'num_heads': num_heads,
        'block_embed_dim': block_embed_dim,
        'backbone_ckpt': BACKBONE_CKPT_PATH,
        'train_data': data_path,
        'valid_split': valid_split,
    }


    model.train()
    optimizer.zero_grad()
    for epoch in range(start_epoch+1, end_epoch+1):
        train_sampler.set_epoch(epoch)
        train_preds = []
        train_labels = []
        total_loss = 0
        total_loss_dist = 0
        total_loss_immune_escape = 0
        total_pos_sum = 0.0
        total_neg_sum = 0.0
        total_pos_count = 0
        total_neg_count = 0
        total_train_n = 0
        lr_sum = 0.0
        lr_count = 0
        accumulation_buffer = []
        for batch_idx, batch in enumerate(train_loader):
            accumulation_buffer.append((batch_idx, batch))

            if len(accumulation_buffer) < accumulation_steps and (batch_idx + 1) < len(train_loader):
                continue

            current_accumulation_steps = len(accumulation_buffer)
            window_n = sum(int(b[4].shape[0]) for _, b in accumulation_buffer)
            for micro_idx, (global_batch_idx, buffered_batch) in enumerate(accumulation_buffer):
                (batch_raw_seqs, batch_mut_seqs, batch_mask_raw, batch_mask_mut,
                 batch_labels, batch_weights, batch_dist_labels,
                 batch_raw_mutations, batch_mut_mutations) = buffered_batch
                sync_context = model.no_sync() if micro_idx < current_accumulation_steps - 1 else nullcontext()
                with sync_context:
                    batch_raw_seqs, batch_mut_seqs = batch_raw_seqs.to(device), batch_mut_seqs.to(device)
                    batch_mask_raw, batch_mask_mut = batch_mask_raw.to(device), batch_mask_mut.to(device)
                    batch_labels,batch_raw_mutations,batch_mut_mutations = batch_labels.to(device),batch_raw_mutations.to(device),batch_mut_mutations.to(device)
                    batch_weights = batch_weights.to(device)
                    batch_dist_labels = batch_dist_labels.to(device)
                    immune_escape_score, avg_l2_per_sample, dist_score = model(
                        batch_raw_seqs, batch_mut_seqs, batch_mask_raw, batch_mask_mut,
                        batch_raw_mutations, batch_mut_mutations)

                    loss_total, batch_stats = compute_total_loss(
                        immune_escape_score,
                        dist_score,
                        batch_labels,
                        batch_weights,
                        batch_dist_labels,
                        avg_l2_per_sample,
                        criterion,
                        cls_threshold,
                    )
                    loss = loss_total * (int(batch_labels.shape[0]) / window_n)
                    loss.backward()

                batch_loss_total = loss_total.item()
                batch_loss_dist = batch_stats['loss_dist']
                batch_loss_immune_escape = batch_stats['loss_immune_escape']
                n_micro = batch_stats['num_samples']
                total_loss += batch_loss_total * n_micro
                total_loss_dist += batch_loss_dist * n_micro
                total_loss_immune_escape += batch_loss_immune_escape * n_micro
                total_pos_sum += batch_stats['avg_l2_pos_sum']
                total_neg_sum += batch_stats['avg_l2_neg_sum']
                total_pos_count += batch_stats['num_pos']
                total_neg_count += batch_stats['num_neg']
                total_train_n += batch_stats['num_samples']

                train_preds.extend(immune_escape_score.cpu().detach().numpy())
                train_labels.extend(batch_labels.float().cpu().numpy())

                if (global_batch_idx+1) % (int(num_samples/world_size)//batch_size//2) == 0:
                    avg_l2_pos = total_pos_sum / total_pos_count if total_pos_count > 0 else float('nan')
                    avg_l2_neg = total_neg_sum / total_neg_count if total_neg_count > 0 else float('nan')
                    corr = spearmanr(train_labels, train_preds)[0]
                    print(f'Epoch {epoch}, Batch {global_batch_idx+1}/{len(train_loader)}, Loss: {batch_loss_total:.4f}, Loss_immune_escape: {batch_loss_immune_escape:.4f}, Loss_dist: {batch_loss_dist:.4f}, Avg L2 pos: {avg_l2_pos:.4f}, Avg L2 neg: {avg_l2_neg:.4f}, Train Spearman correlation: {corr:.4f}',flush=True)

            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            lr_sum += optimizer.param_groups[0]['lr']
            lr_count += 1
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            accumulation_buffer = []


        avg_l2_pos = total_pos_sum / total_pos_count if total_pos_count > 0 else float('nan')
        avg_l2_neg = total_neg_sum / total_neg_count if total_neg_count > 0 else float('nan')
        mean_loss = total_loss / total_train_n if total_train_n > 0 else float('nan')
        corr = spearmanr(train_labels, train_preds)[0]
        mean_loss_dist = total_loss_dist / total_train_n if total_train_n > 0 else float('nan')
        mean_loss_immune_escape = total_loss_immune_escape / total_train_n if total_train_n > 0 else float('nan')
        avg_lr = lr_sum / lr_count if lr_count > 0 else float('nan')
        print(f'Epoch {epoch}, Mean loss: {mean_loss:.6f}, Mean loss immune escape: {mean_loss_immune_escape:.6f}, Mean loss dist: {mean_loss_dist:.6f}, Avg LR: {avg_lr:.6e}, Avg L2 pos: {avg_l2_pos:.4f}, Avg L2 neg: {avg_l2_neg:.4f}, Spearman correlation: {corr:.4f}',flush=True)

        dist.barrier()
        val_rows = evaluate_all_datasets(
            model, val_loaders, criterion, device,
            cls_threshold, rank, world_size,
        )
        if rank == 0:
            for m in val_rows:
                validation_history.append({
                    'epoch': epoch,
                    'dataset': m['dataset'],
                    'loss': m['loss'],
                    'loss_dist': m['loss_dist'],
                    'loss_immune_escape': m['loss_immune_escape'],
                    'spearman_correlation': m['spearman_correlation'],
                    'avg_lr': avg_lr,
                })
            pd.DataFrame(validation_history).round(6).to_csv(result_path, index=False)
            mean_sp = float(np.nanmean([m['spearman_correlation'] for m in val_rows]))
            mean_val_loss = float(np.nanmean([m['loss'] for m in val_rows]))
            print(f"Epoch {epoch}, Validation mean per-dataset Spearman: {mean_sp:.4f}", flush=True)

            os.makedirs('./checkpoints', exist_ok=True)
            ckpt_meta['epoch'] = epoch
            if np.isfinite(mean_val_loss) and mean_val_loss < best_val_loss:
                best_val_loss = mean_val_loss
                p = _ckpt_name('bestvalloss')
                print('Saving best-val-loss checkpoint (epoch {}, {:.6f}) to {}'.format(epoch, mean_val_loss, p), flush=True)
                torch.save({'model_state': model.module.state_dict(), 'meta': ckpt_meta}, p)
            if np.isfinite(mean_sp) and mean_sp > best_val_spearman:
                best_val_spearman = mean_sp
                p = _ckpt_name('bestspearman')
                print('Saving best-spearman checkpoint (epoch {}, {:.4f}) to {}'.format(epoch, mean_sp, p), flush=True)
                torch.save({'model_state': model.module.state_dict(), 'meta': ckpt_meta}, p)
        dist.barrier()

    cleanup()

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--protein', type=str, required=True,
                           help='file-key, e.g. all9 or lovo_ds_sarscov2_rbd')
    argparser.add_argument('--dist_slope_frac', type=float, default=0.1,
                           help='initial effective dist_head slope as a fraction of '
                                'label_std/l2_std (keeps initial dist_score on the label scale)')
    argparser.add_argument('--huber_delta', type=float, default=1.0)
    argparser.add_argument('--grad_clip', type=float, default=1.0)
    argparser.add_argument('--cls_threshold', type=float, default=1.0,
                           help='strong-escape threshold; only used for avg_l2 pos/neg diagnostics')
    argparser.add_argument('--label_col', type=str, default='score_scaled',
                           help='immune escape-head target column; winsorized to '
                                '[--winsor_lo, --winsor_hi] at load time')
    argparser.add_argument('--dist_label_col', '--aux_label_col',
                           dest='dist_label_col', type=str, default='score_scaled',
                           help='auxiliary distance-head target column; winsorized to '
                                '[--winsor_lo, --winsor_hi] at load time')
    argparser.add_argument('--winsor_lo', '--aux_winsor_lo',
                           dest='winsor_lo', type=float, default=-1.5,
                           help='lower winsorization bound for both labels '
                                '(escape bounds from process_dms.py WINSOR)')
    argparser.add_argument('--winsor_hi', '--aux_winsor_hi',
                           dest='winsor_hi', type=float, default=2.5,
                           help='upper winsorization bound for both labels')
    argparser.add_argument('--seed', type=int, default=1000,
                           help='seed for model init and dropout; recorded in checkpoints')
    argparser.add_argument('--valid_datasets', type=str, default='h1n1_ha,h3n2_ha,h5n1_ha,hiv_env,lassa_gp,nipah_rbp,rabies_g,sarscov2_rbd,zika_e',
                           help='comma-separated per-dataset valid files to report Spearman on')
    argparser.add_argument('--epochs', type=int, default=50)
    argparser.add_argument('--valid_split', type=str, default='valid', choices=['valid', 'test'],
                           help="which per-dataset split the validation metrics are computed on "
                                "('test' for LOVO folds: the training datasets' test splits serve "
                                "as the validation set, so selection never sees the held-out dataset)")
    args = argparser.parse_args()

    protein = args.protein
    std_dir = '/home/datasets/cschwang/escape/data/training_data/fitness/DMS_continuous/standard_format'

    print('Processing protein/file-key: {}'.format(protein))
    data_path = os.path.join(std_dir, 'train_{}_independent.csv'.format(protein))
    node_name = os.environ.get('SLURMD_NODENAME', socket.gethostname()).split('.')[0]
    if node_name == 'hkbugpusrv10':
        batch_size = 1
        eval_batchsize = 1
    elif node_name == 'hkbugpudgx01':
        batch_size = 2
        eval_batchsize = 2
    elif node_name in ('hkbugpusrv11', 'hkbugpusrv12', 'hkbugpusrv13',
                       'hkbugpusrv14', 'hkbugpusrv15', 'hkbugpusrv16'):
        batch_size = 4
        eval_batchsize = 4
    else:
        batch_size = 4
        eval_batchsize = 4
    print('Running on node: {}, batch_size: {}, eval_batchsize: {}'.format(node_name, batch_size, eval_batchsize))
    num_layers = 1
    num_heads = 8
    block_embed_dim = 128
    train_model(
        num_layers,
        num_heads,
        block_embed_dim,
        batch_size,
        eval_batchsize,
        data_path,
        protein,
        args.huber_delta,
        args.grad_clip,
        args.cls_threshold,
        args.label_col,
        args.dist_label_col,
        args.valid_datasets.split(','),
        std_dir,
        args.epochs,
        args.valid_split,
        args.dist_slope_frac,
        args.seed,
        (args.winsor_lo, args.winsor_hi),
    )



if __name__ == '__main__':
    main()
