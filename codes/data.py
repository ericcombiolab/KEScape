import torch
from esm.esmfold.v1.misc import batch_encode_sequences
from torch.utils.data import Dataset
import typing as T
from Bio.Align import PairwiseAligner, substitution_matrices

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

def identify_mutations(seq1,seq2):
    aligner = PairwiseAligner()
    aligner.mode = 'global' # Global alignment
    matrix = substitution_matrices.load("BLOSUM62")
    aligner.substitution_matrix = matrix
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -1
    
    alignments = aligner.align(seq1, seq2)
    seq1_aligned = alignments[0][0]
    seq2_aligned = alignments[0][1]

    indel_mutations = [] # we include the two contiguous bases of the indels' boundaries
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

class FitnessDataset(Dataset):
    def __init__(self, raw_seqs, mut_seqs, labels, residue_index_offset: T.Optional[int] = 512, chain_linker: T.Optional[str] = "G" * 25):
        self.raw_seqs = raw_seqs
        self.mut_seqs = mut_seqs
        self.labels = labels
        self.residue_index_offset = residue_index_offset
        self.chain_linker = chain_linker
        self.max_len_raw = max(len(seq) for seq in raw_seqs)
        self.max_len_mut = max(len(seq) for seq in mut_seqs)
    
    def __len__(self):
        return len(self.raw_seqs)
    
    def __getitem__(self, idx):
        raw_seq = [self.raw_seqs[idx]]
        mut_seq = [self.mut_seqs[idx]]
        label = self.labels[idx]
        
        raw_mutations, mut_mutations = identify_mutations(raw_seq[0], mut_seq[0])
        raw_mutations = torch.tensor(raw_mutations)
        mut_mutations = torch.tensor(mut_mutations)
        if raw_mutations.size(0) < self.max_len_raw:
            padding_tensor = torch.full((self.max_len_raw-raw_mutations.size(0), 2), -1)
            raw_mutations = torch.cat((raw_mutations, padding_tensor), dim=0)
        if mut_mutations.size(0) < self.max_len_mut:
            padding_tensor = torch.full((self.max_len_mut-mut_mutations.size(0), 2), -1)
            mut_mutations = torch.cat((mut_mutations, padding_tensor), dim=0)
        
        aatype_raw, mask_raw, _, _, _ = batch_encode_sequences(
            raw_seq, self.residue_index_offset, self.chain_linker
        )
        aatype_raw = aatype_raw[0]
        mask_raw = mask_raw[0]
        
        aatype_mut, mask_mut, _, _, _ = batch_encode_sequences(
            mut_seq, self.residue_index_offset, self.chain_linker
        )
        aatype_mut = aatype_mut[0]
        mask_mut = mask_mut[0]
        
        if len(raw_seq[0]) < self.max_len_raw:
            padding_tensor = torch.zeros(self.max_len_raw-len(raw_seq[0]))
            aatype_raw = torch.cat((aatype_raw, padding_tensor), dim=0)
            mask_raw = torch.cat((mask_raw, padding_tensor), dim=0)
        aatype_raw = aatype_raw.long()
        mask_raw = mask_raw.long()
        
        if len(mut_seq[0]) < self.max_len_mut:
            padding_tensor = torch.zeros(self.max_len_mut-len(mut_seq[0]))
            aatype_mut = torch.cat((aatype_mut, padding_tensor), dim=0)
            mask_mut = torch.cat((mask_mut, padding_tensor), dim=0)
        aatype_mut = aatype_mut.long()
        mask_mut = mask_mut.long()
        
        
        return aatype_raw, aatype_mut, mask_raw, mask_mut, label, raw_mutations, mut_mutations



