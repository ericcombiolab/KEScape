import numpy as np
import torch
from esm.esmfold.v1.misc import batch_encode_sequences
from torch.utils.data import Dataset
import typing as T
from Bio.Align import PairwiseAligner, substitution_matrices

MAX_SEQ_LEN = 1022

def count_seq_gap(i, ele, seq_gap_count):
    if ele == '-':
        if i==0:
            seq_gap_count[i] = 1
        else:
            seq_gap_count[i] = seq_gap_count[i-1] + 1
    else:
        if i==0:
            seq_gap_count[i] = 0
        else:
            seq_gap_count[i] = seq_gap_count[i-1]
    return seq_gap_count

def _get_aligner():
    aligner = PairwiseAligner()
    aligner.mode = 'global'
    matrix = substitution_matrices.load("BLOSUM62")
    aligner.substitution_matrix = matrix
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -1
    return aligner

def _mutations_from_alignment(seq1_aligned, seq2_aligned):
    indel_mutations = []
    point_mutations = []
    seq1_gap_count = {}
    seq2_gap_count = {}
    for i in range(len(seq1_aligned)):
        ele1 = seq1_aligned[i]
        ele2 = seq2_aligned[i]
        seq1_gap_count = count_seq_gap(i, ele1, seq1_gap_count)
        seq2_gap_count = count_seq_gap(i, ele2, seq2_gap_count)

        if ele1 != ele2:
            if ele1 == '-' or ele2 == '-':
                if len(indel_mutations) == 0:
                    indel_mutations.append([max(0,i-1),min(len(seq1_aligned)-1,i+1)])
                else:
                    if indel_mutations[-1][1] == i:
                        indel_mutations[-1][1] = min(len(seq1_aligned)-1,i+1)
                    else:
                        indel_mutations.append([max(0,i-1),min(len(seq1_aligned)-1,i+1)])
            else:
                point_mutations.append([i, i])

    mutations = indel_mutations + point_mutations
    seq1_mutations = []
    seq2_mutations = []

    for i in range(len(mutations)):
        start = mutations[i][0]
        end = mutations[i][1]
        if start == 0:
            gap_count1_start = 0
            gap_count2_start = 0
        else:
            gap_count1_start = seq1_gap_count[start-1]
            gap_count2_start = seq2_gap_count[start-1]
        gap_count1_end = seq1_gap_count[end]
        gap_count2_end = seq2_gap_count[end]
        seq1_mutations.append([start-gap_count1_start, end-gap_count1_end])
        seq2_mutations.append([start-gap_count2_start, end-gap_count2_end])

    return seq1_mutations, seq2_mutations

def identify_mutations(seq1,seq2):
    aligner = _get_aligner()
    alignments = aligner.align(seq1, seq2)
    return _mutations_from_alignment(alignments[0][0], alignments[0][1])

def truncate_and_identify(seq1, seq2, max_len):
    aligner = _get_aligner()
    alignments = aligner.align(seq1, seq2)
    seq1_aligned = alignments[0][0]
    seq2_aligned = alignments[0][1]
    n_cols = len(seq1_aligned)

    mut_cols = [i for i, (a, b) in enumerate(zip(seq1_aligned, seq2_aligned)) if a != b]
    if len(mut_cols) == 0:
        c0, c1 = 0, min(max_len, n_cols)
    else:
        first, last = mut_cols[0], mut_cols[-1]
        span = last - first + 1
        if span >= max_len:
            c0 = first
        else:
            c0 = (first + last + 1 - max_len) // 2
            c0 = max(0, min(c0, n_cols - max_len))
        c1 = min(c0 + max_len, n_cols)

    sub1_aligned = seq1_aligned[c0:c1]
    sub2_aligned = seq2_aligned[c0:c1]
    seq1_mutations, seq2_mutations = _mutations_from_alignment(sub1_aligned, sub2_aligned)
    sub1 = sub1_aligned.replace('-', '')
    sub2 = sub2_aligned.replace('-', '')
    return sub1, sub2, seq1_mutations, seq2_mutations

def _encode_pair(raw_seq_str, mut_seq_str,
                 residue_index_offset=512, chain_linker="G" * 25):
    if max(len(raw_seq_str), len(mut_seq_str)) > MAX_SEQ_LEN:
        raw_seq_str, mut_seq_str, raw_mutations, mut_mutations = truncate_and_identify(
            raw_seq_str, mut_seq_str, MAX_SEQ_LEN
        )
    else:
        raw_mutations, mut_mutations = identify_mutations(raw_seq_str, mut_seq_str)

    aatype_raw, mask_raw, _, _, _ = batch_encode_sequences(
        [raw_seq_str], residue_index_offset, chain_linker
    )
    aatype_mut, mask_mut, _, _, _ = batch_encode_sequences(
        [mut_seq_str], residue_index_offset, chain_linker
    )
    return (
        aatype_raw[0].numpy().astype(np.int16),
        mask_raw[0].numpy().astype(np.int8),
        aatype_mut[0].numpy().astype(np.int16),
        mask_mut[0].numpy().astype(np.int8),
        np.asarray(raw_mutations, dtype=np.int16),
        np.asarray(mut_mutations, dtype=np.int16),
    )

def _encode_pair_star(args):
    return _encode_pair(*args)

def collate_variable_length(batch):
    n = len(batch)
    n_fields = len(batch[0])
    aatype_raw = [b[0] for b in batch]
    aatype_mut = [b[1] for b in batch]
    mask_raw = [b[2] for b in batch]
    mask_mut = [b[3] for b in batch]
    raw_mutations = [b[-2] for b in batch]
    mut_mutations = [b[-1] for b in batch]

    len_raw = max(t.numel() for t in aatype_raw)
    len_mut = max(t.numel() for t in aatype_mut)
    max_mut_raw = max(1, max(t.size(0) for t in raw_mutations))
    max_mut_mut = max(1, max(t.size(0) for t in mut_mutations))

    ar = aatype_raw[0].new_zeros((n, len_raw))
    am = aatype_mut[0].new_zeros((n, len_mut))
    mr = mask_raw[0].new_zeros((n, len_raw))
    mm = mask_mut[0].new_zeros((n, len_mut))
    rmut = raw_mutations[0].new_full((n, max_mut_raw, 2), -1)
    mmut = mut_mutations[0].new_full((n, max_mut_mut, 2), -1)
    for i in range(n):
        lr = aatype_raw[i].numel()
        lm = aatype_mut[i].numel()
        ar[i, :lr] = aatype_raw[i]
        mr[i, :lr] = mask_raw[i]
        am[i, :lm] = aatype_mut[i]
        mm[i, :lm] = mask_mut[i]
        nr = raw_mutations[i].size(0)
        nm = mut_mutations[i].size(0)
        if nr > 0:
            rmut[i, :nr] = raw_mutations[i]
        if nm > 0:
            mmut[i, :nm] = mut_mutations[i]

    labels = torch.stack([torch.as_tensor(b[4]) for b in batch])
    if n_fields == 7:
        return ar, am, mr, mm, labels, rmut, mmut
    weights = torch.stack([torch.as_tensor(b[5]) for b in batch])
    dist_labels = torch.stack([torch.as_tensor(b[6]) for b in batch])
    return ar, am, mr, mm, labels, weights, dist_labels, rmut, mmut

class FitnessDataset(Dataset):
    def __init__(self, raw_seqs, mut_seqs, labels, residue_index_offset: T.Optional[int] = 512,
                 chain_linker: T.Optional[str] = "G" * 25, nprocs: int = 1,
                 items=None):
        self.raw_seqs = raw_seqs
        self.mut_seqs = mut_seqs
        self.labels = labels
        self.residue_index_offset = residue_index_offset
        self.chain_linker = chain_linker
        self.max_len_raw = max(len(seq) for seq in raw_seqs)
        self.max_len_mut = max(len(seq) for seq in mut_seqs)
        if max(self.max_len_raw, self.max_len_mut) > MAX_SEQ_LEN:
            print('FitnessDataset: sequence length ({}/{}) exceeds the model limit {}; '
                  'applying mutation-aware truncation to {}.'.format(
                      self.max_len_raw, self.max_len_mut, MAX_SEQ_LEN, MAX_SEQ_LEN))
            self.max_len_raw = min(self.max_len_raw, MAX_SEQ_LEN)
            self.max_len_mut = min(self.max_len_mut, MAX_SEQ_LEN)

        self._items = items
        if self._items is not None:
            assert len(self._items) == len(self.raw_seqs), \
                'items must be aligned with raw_seqs'
        elif nprocs and nprocs > 0 and len(self.raw_seqs) > 0:
            if nprocs > 1:
                import multiprocessing as mp
                with mp.get_context('fork').Pool(nprocs) as pool:
                    self._items = pool.map(
                        _encode_pair_star,
                        zip(self.raw_seqs, self.mut_seqs,
                            [self.residue_index_offset] * len(self.raw_seqs),
                            [self.chain_linker] * len(self.raw_seqs)),
                        chunksize=64,
                    )
            else:
                self._items = [
                    _encode_pair(r, m, self.residue_index_offset, self.chain_linker)
                    for r, m in zip(self.raw_seqs, self.mut_seqs)
                ]

    def __len__(self):
        return len(self.raw_seqs)

    def __getitem__(self, idx):
        if self._items is None:
            (aatype_raw, mask_raw, aatype_mut, mask_mut,
             raw_mutations, mut_mutations) = _encode_pair(
                self.raw_seqs[idx], self.mut_seqs[idx],
                self.residue_index_offset, self.chain_linker)
        else:
            (aatype_raw, mask_raw, aatype_mut, mask_mut,
             raw_mutations, mut_mutations) = self._items[idx]

        return (torch.as_tensor(aatype_raw).long(), torch.as_tensor(aatype_mut).long(),
                torch.as_tensor(mask_raw).long(), torch.as_tensor(mask_mut).long(),
                self.labels[idx], torch.as_tensor(raw_mutations).long(),
                torch.as_tensor(mut_mutations).long())

class PooledFitnessDataset(FitnessDataset):
    def __init__(self, raw_seqs, mut_seqs, labels, weights, dist_labels, **kw):
        super().__init__(raw_seqs, mut_seqs, labels, **kw)
        assert len(weights) == len(raw_seqs) and len(dist_labels) == len(raw_seqs)
        self.weights = weights
        self.dist_labels = dist_labels

    def __getitem__(self, idx):
        aatype_raw, aatype_mut, mask_raw, mask_mut, label, raw_mut, mut_mut = \
            super().__getitem__(idx)
        return (aatype_raw, aatype_mut, mask_raw, mask_mut, label,
                float(self.weights[idx]), float(self.dist_labels[idx]),
                raw_mut, mut_mut)
