import pandas as pd
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import spearmanr
import esm
import math
from model import FitnessPredictor
from data import FitnessDataset
from datetime import timedelta
import argparse
from contextlib import nullcontext

def setup():
    # torchrun sets MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE, LOCAL_RANK.
    dist.init_process_group(backend='nccl', init_method='env://', timeout=timedelta(hours=1))

def cleanup():
    dist.destroy_process_group()

def load_data(rank,world_size,data_path,batch_size,is_train):
    if is_train:
        df = pd.read_csv('{}'.format(data_path))
        raw_seq_arr_train = df.loc[:,'raw_seq'].values
        mut_seq_arr_train = df.loc[:,'mut_seq'].values
        judge_same = (raw_seq_arr_train == mut_seq_arr_train)
        if sum(judge_same) > 0:
            print('There are {} samples with raw_seq == mut_seq'.format(sum(judge_same)))
            raise Exception('There are samples with raw_seq == mut_seq')
        score_train = df.loc[:,'label'].values
        train_dataset = FitnessDataset(raw_seq_arr_train,mut_seq_arr_train,score_train)
        sampler = DistributedSampler(train_dataset,num_replicas=world_size,rank=rank,shuffle=True)
        loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, sampler=sampler)
    else:
        df = pd.read_csv('{}'.format(data_path))
        raw_seq_arr_test = df.loc[:,'raw_seq'].values
        mut_seq_arr_test = df.loc[:,'mut_seq'].values
        score_test = df.loc[:,'label'].values
        val_dataset = FitnessDataset(raw_seq_arr_test,mut_seq_arr_test,score_test)
        sampler = DistributedSampler(val_dataset,num_replicas=world_size,rank=rank)
        loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, sampler=sampler)
    return sampler, loader, len(df)

def load_validation_data(data_path, batch_size):
    df = pd.read_csv('{}'.format(data_path))
    raw_seq_arr_val = df.loc[:,'raw_seq'].values
    mut_seq_arr_val = df.loc[:,'mut_seq'].values
    score_val = df.loc[:,'label'].values
    val_dataset = FitnessDataset(raw_seq_arr_val, mut_seq_arr_val, score_val)
    loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return loader

def compute_l2_components(
    avg_l2_per_sample,
    labels,
    pos_margin,
    neg_margin,
    l2_loss_ceiling,
    l2_pos_weight=1.0,
    l2_neg_weight=1.0,
):
    zero = avg_l2_per_sample.new_tensor(0.0)
    pos_mask = labels == 1
    neg_mask = labels == 0

    if pos_mask.any():
        pos_penalty = torch.clamp(F.relu(pos_margin - avg_l2_per_sample[pos_mask]), max=l2_loss_ceiling)
        pos_penalty_sum = pos_penalty.sum()
        loss_pos = pos_penalty.mean()
    else:
        pos_penalty_sum = zero
        loss_pos = zero

    if neg_mask.any():
        neg_penalty = torch.clamp(F.relu(avg_l2_per_sample[neg_mask] - neg_margin), max=l2_loss_ceiling)
        neg_penalty_sum = neg_penalty.sum()
        loss_neg = neg_penalty.mean()
    else:
        neg_penalty_sum = zero
        loss_neg = zero

    num_pos = int(pos_mask.sum().item())
    num_neg = int(neg_mask.sum().item())
    weighted_penalty_sum = l2_pos_weight * pos_penalty_sum + l2_neg_weight * neg_penalty_sum
    total_sample_weight = l2_pos_weight * num_pos + l2_neg_weight * num_neg
    loss_l2 = weighted_penalty_sum / total_sample_weight if total_sample_weight > 0 else zero

    return {
        'loss_l2': loss_l2,
        'loss_pos': loss_pos,
        'loss_neg': loss_neg,
        'pos_mask': pos_mask,
        'neg_mask': neg_mask,
        'weighted_penalty_sum': weighted_penalty_sum,
        'total_sample_weight': total_sample_weight,
        'num_pos': num_pos,
        'num_neg': num_neg,
    }

def compute_l2_loss(
    avg_l2_per_sample,
    labels,
    pos_margin,
    neg_margin,
    l2_loss_ceiling,
    l2_pos_weight=1.0,
    l2_neg_weight=1.0,
):
    l2_components = compute_l2_components(
        avg_l2_per_sample,
        labels,
        pos_margin,
        neg_margin,
        l2_loss_ceiling,
        l2_pos_weight,
        l2_neg_weight,
    )
    return (
        l2_components['loss_l2'],
        l2_components['loss_pos'],
        l2_components['loss_neg'],
        l2_components['pos_mask'],
        l2_components['neg_mask'],
    )

def compute_total_loss(
    logits,
    labels,
    avg_l2_per_sample,
    criterion,
    lambda_l2,
    pos_margin,
    neg_margin,
    l2_loss_ceiling,
    l2_pos_weight=1.0,
    l2_neg_weight=1.0,
):
    loss_cls = criterion(logits, labels)
    l2_components = compute_l2_components(
        avg_l2_per_sample,
        labels,
        pos_margin,
        neg_margin,
        l2_loss_ceiling,
        l2_pos_weight,
        l2_neg_weight,
    )
    loss_l2 = l2_components['loss_l2']
    loss_pos = l2_components['loss_pos']
    loss_neg = l2_components['loss_neg']
    pos_mask = l2_components['pos_mask']
    neg_mask = l2_components['neg_mask']
    loss = loss_cls + lambda_l2 * loss_l2

    batch_stats = {
        'loss_cls': loss_cls.item(),
        'loss_l2': loss_l2.item(),
        'loss_pos': loss_pos.item(),
        'loss_neg': loss_neg.item(),
        'avg_l2_pos_sum': avg_l2_per_sample[pos_mask].sum().item() if pos_mask.any() else 0.0,
        'avg_l2_neg_sum': avg_l2_per_sample[neg_mask].sum().item() if neg_mask.any() else 0.0,
        'num_pos': l2_components['num_pos'],
        'num_neg': l2_components['num_neg'],
    }
    return loss, batch_stats

def evaluate_model(
    model,
    val_loader,
    criterion,
    device,
    lambda_l2,
    pos_margin,
    neg_margin,
    l2_loss_ceiling,
    l2_pos_weight=1.0,
    l2_neg_weight=1.0,
):
    model.eval()
    val_preds = []
    val_labels = []
    total_loss = 0.0
    total_cls_loss = 0.0
    total_l2_loss = 0.0
    total_pos_sum = 0.0
    total_neg_sum = 0.0
    total_pos_count = 0
    total_neg_count = 0

    with torch.no_grad():
        for batch_raw_seqs, batch_mut_seqs, batch_mask_raw, batch_mask_mut, batch_labels, batch_raw_mutations, batch_mut_mutations in val_loader:
            batch_raw_seqs = batch_raw_seqs.to(device)
            batch_mut_seqs = batch_mut_seqs.to(device)
            batch_mask_raw = batch_mask_raw.to(device)
            batch_mask_mut = batch_mask_mut.to(device)
            batch_labels = batch_labels.to(device)
            batch_raw_mutations = batch_raw_mutations.to(device)
            batch_mut_mutations = batch_mut_mutations.to(device)

            _, output, avg_l2_per_sample = model(batch_raw_seqs, batch_mut_seqs, batch_mask_raw, batch_mask_mut, batch_raw_mutations, batch_mut_mutations)
            loss, batch_stats = compute_total_loss(
                output,
                batch_labels,
                avg_l2_per_sample,
                criterion,
                lambda_l2,
                pos_margin,
                neg_margin,
                l2_loss_ceiling,
                l2_pos_weight,
                l2_neg_weight,
            )
            total_loss += loss.item()
            total_cls_loss += batch_stats['loss_cls']
            total_l2_loss += batch_stats['loss_l2']
            total_pos_sum += batch_stats['avg_l2_pos_sum']
            total_neg_sum += batch_stats['avg_l2_neg_sum']
            total_pos_count += batch_stats['num_pos']
            total_neg_count += batch_stats['num_neg']

            val_preds.extend(output[:, 1].detach().cpu().numpy())
            val_labels.extend(batch_labels.float().cpu().numpy())

    mean_loss = total_loss / len(val_loader)
    metrics = {
        'loss': mean_loss,
        'loss_cls': total_cls_loss / len(val_loader),
        'loss_l2': total_l2_loss / len(val_loader),
        'auroc': np.nan,
        'auprc': np.nan,
        'spearman_correlation': spearmanr(val_labels, val_preds)[0],
        'avg_l2_pos': total_pos_sum / total_pos_count if total_pos_count > 0 else np.nan,
        'avg_l2_neg': total_neg_sum / total_neg_count if total_neg_count > 0 else np.nan,
    }
    if len(set(val_labels)) == 2:
        metrics['auroc'] = roc_auc_score(val_labels, val_preds)
        metrics['auprc'] = average_precision_score(val_labels, val_preds)

    model.train()
    return metrics

def train_model(
    num_layers,
    num_heads,
    block_embed_dim,
    batch_size,
    eval_batchsize,
    data_path,
    val_data_path,
    protein,
    ratio,
    lambda_l2,
    pos_margin,
    neg_margin,
    l2_loss_ceiling,
    l2_pos_weight,
    l2_neg_weight,
):
    setup()

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    
    train_sampler, train_loader, num_samples = load_data(rank,world_size,data_path,batch_size,is_train=True)
    
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    
    model = FitnessPredictor(num_layers,num_heads,block_embed_dim).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    
    lr_params = 1e-3  # Generic learning rate for other parameters
    # Create parameter groups
    param_groups = [
        {"params": [p for n, p in model.named_parameters()], "lr": lr_params}
    ]
    
    val_loader = load_validation_data(val_data_path, eval_batchsize) if rank == 0 else None
    validation_history = []
    result_path = f'./model_validation_performance.csv'
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([ratio,1]).to(device))
    if l2_pos_weight is None:
        l2_pos_weight = 1.0 / ratio
    if l2_neg_weight is None:
        l2_neg_weight = 1.0

    optimizer = optim.AdamW(param_groups, betas= (0.9,0.98), weight_decay=0.01)
    
    
    start_epoch = 0
    end_epoch = 30
    num_epochs = end_epoch - start_epoch
    accumulation_batch_size = 128 # accumulate gradients with a batch size of accumulation_batch_size
    accumulation_steps = accumulation_batch_size // (batch_size*world_size)
    
    batches_per_epoch = int(np.ceil(num_samples/world_size/batch_size))
    steps_per_epoch = batches_per_epoch // accumulation_steps
    # Add one more step per epoch if there are leftover batches
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
    
    
    model.train()
    optimizer.zero_grad()
    for epoch in range(start_epoch+1, end_epoch+1):
        train_sampler.set_epoch(epoch)
        train_preds = []
        train_labels = []
        total_loss = 0
        total_loss_cls = 0
        total_loss_l2 = 0
        total_pos_sum = 0.0
        total_neg_sum = 0.0
        total_pos_count = 0
        total_neg_count = 0
        accumulation_buffer = []
        for batch_idx, batch in enumerate(train_loader):
            accumulation_buffer.append((batch_idx, batch))

            if len(accumulation_buffer) < accumulation_steps and (batch_idx + 1) < len(train_loader):
                continue

            current_accumulation_steps = len(accumulation_buffer)
            window_pos_count = 0
            window_neg_count = 0
            for _, buffered_batch in accumulation_buffer:
                buffered_labels = buffered_batch[4]
                window_pos_count += int((buffered_labels == 1).sum().item())
                window_neg_count += int((buffered_labels == 0).sum().item())
            # Use one denominator for the whole accumulation window so the L2
            # term matches the effective batch instead of each microbatch.
            window_l2_denominator = l2_pos_weight * window_pos_count + l2_neg_weight * window_neg_count

            for micro_idx, (global_batch_idx, buffered_batch) in enumerate(accumulation_buffer):
                batch_raw_seqs, batch_mut_seqs, batch_mask_raw, batch_mask_mut, batch_labels, batch_raw_mutations, batch_mut_mutations = buffered_batch
                sync_context = model.no_sync() if micro_idx < current_accumulation_steps - 1 else nullcontext()
                with sync_context:
                    batch_raw_seqs, batch_mut_seqs = batch_raw_seqs.to(device), batch_mut_seqs.to(device)
                    batch_mask_raw, batch_mask_mut = batch_mask_raw.to(device), batch_mask_mut.to(device)
                    batch_labels,batch_raw_mutations,batch_mut_mutations = batch_labels.to(device),batch_raw_mutations.to(device),batch_mut_mutations.to(device)
                    _, output, avg_l2_per_sample = model(batch_raw_seqs,batch_mut_seqs,batch_mask_raw,batch_mask_mut,batch_raw_mutations,batch_mut_mutations)

                    loss_cls = criterion(output, batch_labels)
                    l2_components = compute_l2_components(
                        avg_l2_per_sample,
                        batch_labels,
                        pos_margin,
                        neg_margin,
                        l2_loss_ceiling,
                        l2_pos_weight,
                        l2_neg_weight,
                    )
                    loss_l2_window = (
                        l2_components['weighted_penalty_sum'] / window_l2_denominator
                        if window_l2_denominator > 0
                        else avg_l2_per_sample.new_tensor(0.0)
                    )
                    loss = (loss_cls / current_accumulation_steps) + lambda_l2 * loss_l2_window
                    loss.backward()

                batch_loss_l2 = l2_components['loss_l2'].item()
                batch_loss_total = loss_cls.item() + lambda_l2 * batch_loss_l2
                batch_pos_sum = avg_l2_per_sample[l2_components['pos_mask']].sum().item() if l2_components['num_pos'] > 0 else 0.0
                batch_neg_sum = avg_l2_per_sample[l2_components['neg_mask']].sum().item() if l2_components['num_neg'] > 0 else 0.0
                total_loss += batch_loss_total
                total_loss_cls += loss_cls.item()
                total_loss_l2 += batch_loss_l2
                total_pos_sum += batch_pos_sum
                total_neg_sum += batch_neg_sum
                total_pos_count += l2_components['num_pos']
                total_neg_count += l2_components['num_neg']

                output_np = output[:,1].cpu().detach().numpy()
                batch_labels_np = batch_labels.float().cpu().numpy()

                train_preds.extend(output_np)
                train_labels.extend(batch_labels_np)

                if (global_batch_idx+1) % (int(num_samples/world_size)//batch_size//2) == 0:
                    avg_l2_pos = total_pos_sum / total_pos_count if total_pos_count > 0 else float('nan')
                    avg_l2_neg = total_neg_sum / total_neg_count if total_neg_count > 0 else float('nan')
                    if len(set(train_labels)) == 2:
                        train_auroc = roc_auc_score(train_labels, train_preds)
                        train_auprc = average_precision_score(train_labels, train_preds)
                        corr = spearmanr(train_labels, train_preds)[0]
                        print(f'Epoch {epoch}, Batch {global_batch_idx+1}/{len(train_loader)}, Loss: {batch_loss_total:.4f}, Loss_cls: {loss_cls.item():.4f}, Loss_l2: {batch_loss_l2:.4f}, Avg L2 pos: {avg_l2_pos:.4f}, Avg L2 neg: {avg_l2_neg:.4f}, Train AUROC: {train_auroc:.4f}, Train AUPRC: {train_auprc:.4f}, Train Spearmenn correlation: {corr:.4f}',flush=True)
                    else:
                        corr = spearmanr(train_labels, train_preds)[0]
                        print(f'Epoch {epoch}, Batch {global_batch_idx+1}/{len(train_loader)}, Loss: {batch_loss_total:.4f}, Loss_cls: {loss_cls.item():.4f}, Loss_l2: {batch_loss_l2:.4f}, Avg L2 pos: {avg_l2_pos:.4f}, Avg L2 neg: {avg_l2_neg:.4f}, Train Spearmann correlation: {corr:.4f}',flush=True)

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            accumulation_buffer = []

        
        avg_l2_pos = total_pos_sum / total_pos_count if total_pos_count > 0 else float('nan')
        avg_l2_neg = total_neg_sum / total_neg_count if total_neg_count > 0 else float('nan')
        if len(set(train_labels)) == 2:
            train_auroc = roc_auc_score(train_labels, train_preds)
            train_auprc = average_precision_score(train_labels, train_preds)
            corr = spearmanr(train_labels, train_preds)[0]
            mean_loss = total_loss/len(train_loader)
            mean_loss_cls = total_loss_cls / len(train_loader)
            mean_loss_l2 = total_loss_l2 / len(train_loader)
            print(f'Epoch {epoch}, Mean loss: {mean_loss:.6f}, Mean loss cls: {mean_loss_cls:.6f}, Mean loss l2: {mean_loss_l2:.6f}, Avg L2 pos: {avg_l2_pos:.4f}, Avg L2 neg: {avg_l2_neg:.4f}, Train AUROC: {train_auroc:.4f}, Train AUPRC: {train_auprc:.4f}, Spearmann correlation: {corr:.4f}',flush=True)
        else:
            mean_loss = total_loss/len(train_loader)
            corr = spearmanr(train_labels, train_preds)[0]
            mean_loss_cls = total_loss_cls / len(train_loader)
            mean_loss_l2 = total_loss_l2 / len(train_loader)
            print(f'Epoch {epoch}, Mean loss: {mean_loss:.6f}, Mean loss cls: {mean_loss_cls:.6f}, Mean loss l2: {mean_loss_l2:.6f}, Avg L2 pos: {avg_l2_pos:.4f}, Avg L2 neg: {avg_l2_neg:.4f}, Spearmann correlation: {corr:.4f}',flush=True)
        
        dist.barrier()
        if rank == 0:
            val_metrics = evaluate_model(
                model.module,
                val_loader,
                criterion,
                device,
                lambda_l2,
                pos_margin,
                neg_margin,
                l2_loss_ceiling,
                l2_pos_weight,
                l2_neg_weight,
            )
            validation_history.append({
                'epoch': epoch,
                'loss': val_metrics['loss'],
                'loss_cls': val_metrics['loss_cls'],
                'loss_l2': val_metrics['loss_l2'],
                'AUROC': val_metrics['auroc'],
                'AUPRC': val_metrics['auprc'],
                'spearman_correlation': val_metrics['spearman_correlation'],
                'avg_l2_pos': val_metrics['avg_l2_pos'],
                'avg_l2_neg': val_metrics['avg_l2_neg'],
            })
            pd.DataFrame(validation_history).to_csv(result_path, index=False)
            print(
                f"Epoch {epoch}, Validation loss: {val_metrics['loss']:.6f}, "
                f"Validation loss cls: {val_metrics['loss_cls']:.6f}, "
                f"Validation loss l2: {val_metrics['loss_l2']:.6f}, "
                f"Validation avg L2 pos: {val_metrics['avg_l2_pos']:.4f}, "
                f"Validation avg L2 neg: {val_metrics['avg_l2_neg']:.4f}, "
                f"Validation AUROC: {val_metrics['auroc']:.4f}, "
                f"Validation AUPRC: {val_metrics['auprc']:.4f}, "
                f"Validation Spearmann correlation: {val_metrics['spearman_correlation']:.4f}",
                flush=True
            )
        dist.barrier()
        
        model_path = './model_epoch{}_{}_lambdal2{}.pt'.format(epoch,protein,lambda_l2)
        if dist.get_rank() == 0:
            print('Saving model to {}'.format(model_path))
            torch.save(model.module.state_dict(), model_path)
    
    cleanup()

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--protein', type=str, required=True)
    argparser.add_argument('--lambda_l2', type=float, default=0.05)
    argparser.add_argument('--pos_margin', type=float, default=40.0)
    argparser.add_argument('--neg_margin', type=float, default=20.0)
    argparser.add_argument('--l2_loss_ceiling', type=float, default=5)
    argparser.add_argument('--l2_pos_weight', type=float, default=None)
    argparser.add_argument('--l2_neg_weight', type=float, default=None)
    args = argparser.parse_args()
    
    protein = args.protein
    
    ratio_dic = {'spike_sars2':0.2008,
                 'ZIKA_envelope':0.0404,
                 'Nipah_RBP':0.0361,
                 'HIV_envelope':0.0062,
                 'SARS2_XBB.1.5_spike':0.0412,
                 'lassa_GP':0.0286,
                 'H3N2_HA':0.0198,
                 'H3N2_NA':0.0545,
                 'H5_HA':0.035,
                 'H1_HA':0.0155,
                 'rabies_glyco':0.0272} # This dictionary is just a sample for demonstration. You should replace it with the actual ratio for your cases.
    ratio = ratio_dic[protein]

    print('Processing protein: {}'.format(protein))
    data_path = '/home/datasets/cschwang/escape/data/training_data/fitness/DMS/standard_format/train_{}_gamma_pairs_no_aug_independent.csv'.format(protein)
    val_data_path = '/home/datasets/cschwang/escape/data/training_data/fitness/DMS/standard_format/valid_{}_gamma_pairs_no_aug_independent.csv'.format(protein)
    batch_size = 1
    eval_batchsize = 1
    num_layers = 1 # number of Transformer encoder layers added after ESM model
    num_heads = 8
    block_embed_dim = 128
    train_model(
        num_layers,
        num_heads,
        block_embed_dim,
        batch_size,
        eval_batchsize,
        data_path,
        val_data_path,
        protein,
        ratio,
        args.lambda_l2,
        args.pos_margin,
        args.neg_margin,
        args.l2_loss_ceiling,
        args.l2_pos_weight,
        args.l2_neg_weight,
    )
    
    

if __name__ == '__main__':
    main()
    
            
            
    


