import torch

def fetch_local_embed_noboseos(seq_embed, structure_embed, seq_mutations):
    # This function is to fetch the local structure embeddings based on the mutations
    max_mut_num = 0 # maximum number of mutations in the batch
    for batch in range(structure_embed.size(0)):
        tmp_max = 0
        for mutation in seq_mutations[batch]:
            if mutation[0] == -1:
                break
            tmp_max += 1
        if tmp_max > max_mut_num:
            max_mut_num = tmp_max
    seq_embed_local = torch.zeros(seq_embed.size(0), max_mut_num, seq_embed.size(2)).to(seq_embed.device)
    structure_embed_local = torch.zeros(structure_embed.size(0), max_mut_num, structure_embed.size(2)).to(structure_embed.device)
    for batch in range(structure_embed.size(0)):
        sub_seq_embed = seq_embed[batch]
        sub_structure_embed = structure_embed[batch]
        for i, mutation in enumerate(seq_mutations[batch]):
            start_index = mutation[0]
            if start_index == -1:
                break
            end_index = mutation[1]
            if start_index == end_index:
                seq_embed_local[batch, i] = sub_seq_embed[start_index]
                structure_embed_local[batch, i] = sub_structure_embed[start_index]
            else:
                seq_embed_local[batch, i] = sub_seq_embed[start_index:end_index+1].mean(dim=0)
                structure_embed_local[batch, i] = sub_structure_embed[start_index:end_index+1].mean(dim=0)
    # check whether all values are 0
    if torch.all(seq_embed_local.eq(0)) or torch.all(structure_embed_local.eq(0)):
        raise Exception('Error: No mutation found in the sequence')
    return seq_embed_local, structure_embed_local

def fetch_local_embed(seq_embed, structure_embed, seq_mutations):
    # This function is to fetch the local structure embeddings based on the mutations
    max_mut_num = 0 # maximum number of mutations in the batch
    for batch in range(structure_embed.size(0)):
        tmp_max = 0
        for mutation in seq_mutations[batch]:
            if mutation[0] == -1:
                break
            tmp_max += 1
        if tmp_max > max_mut_num:
            max_mut_num = tmp_max
    seq_embed_local = torch.zeros(seq_embed.size(0), max_mut_num, seq_embed.size(2)).to(seq_embed.device)
    structure_embed_local = torch.zeros(structure_embed.size(0), max_mut_num, structure_embed.size(2)).to(structure_embed.device)
    for batch in range(structure_embed.size(0)):
        sub_seq_embed = seq_embed[batch]
        sub_structure_embed = structure_embed[batch]
        for i, mutation in enumerate(seq_mutations[batch]):
            start_index = mutation[0]
            if start_index == -1:
                break
            end_index = mutation[1]
            if start_index == end_index:
                seq_embed_local[batch, i] = sub_seq_embed[start_index+1] # for seq_embed we plus 1 as it contains bos and eos
                structure_embed_local[batch, i] = sub_structure_embed[start_index]
            else:
                seq_embed_local[batch, i] = sub_seq_embed[start_index+1:end_index+2].mean(dim=0)
                structure_embed_local[batch, i] = sub_structure_embed[start_index:end_index+1].mean(dim=0)
    # check whether all values are 0
    if torch.all(seq_embed_local.eq(0)) or torch.all(structure_embed_local.eq(0)):
        raise Exception('Error: No mutation found in the sequence')
    return seq_embed_local, structure_embed_local

def fetch_pairwise_diff_single(residues,masked_contact_repre,batch_indices,batch_size, num_mutations, seq_len,padding_mask):
    padding_tensor = padding_mask.unsqueeze(1).expand(-1, num_mutations, -1)
    pairwise_residues = masked_contact_repre[batch_indices.flatten(), residues.flatten()].view(batch_size, num_mutations, seq_len)
    residue_vals = pairwise_residues.gather(2, residues.unsqueeze(2))
    pairwise_diff = ((pairwise_residues - residue_vals) ** 2 * padding_tensor).sum(dim=2) / (seq_len-1)
    return pairwise_diff

def fetch_pairwise_diff_multiple(residues,masked_contact_repre,batch_indices,batch_size, num_mutations, seq_len,padding_mask):
    padding_tensor = padding_mask[batch_indices.flatten()].unsqueeze(1).view(batch_size, num_mutations, seq_len)
    pairwise_residues = masked_contact_repre[batch_indices.flatten(), residues.flatten()].view(batch_size, num_mutations, seq_len)
    residue_vals = pairwise_residues.gather(2, residues.unsqueeze(2))
    pairwise_diff = ((pairwise_residues - residue_vals) ** 2 * padding_tensor).sum(dim=2) / (seq_len-1)
    return pairwise_diff

def fetch_local_contact(contact_repre,seq_mutations,padding_mask):
    padding_mask = ~padding_mask
    batch_size, seq_len = contact_repre.size(0), contact_repre.size(1)
    num_mutations = seq_mutations.size(1)
    device = contact_repre.device
    
    # Expand padding mask
    padding_expand = padding_mask.unsqueeze(1) & padding_mask.unsqueeze(2)
    masked_contact_repre = contact_repre*padding_expand
    
    batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, num_mutations)
    mutation_start = seq_mutations[:, :, 0]
    mutation_end = seq_mutations[:, :, 1]
    
    # identify padding elements in seq_mutations (value with -1)
    mutation_padding_indices = (mutation_start == -1)
    mutation_start[mutation_padding_indices] = seq_len-1 # change the value of -1
    
    single_mutation_mask = (mutation_start == mutation_end)
    single_mutation_diffs = fetch_pairwise_diff_single(mutation_start,masked_contact_repre,batch_indices,batch_size, num_mutations, seq_len, padding_mask)
    
    multiple_mutation_mask = ~single_mutation_mask & ~mutation_padding_indices
    multiple_start = mutation_start[multiple_mutation_mask]
    multiple_end = mutation_end[multiple_mutation_mask]
    
    if multiple_mutation_mask.any():
        max_range = (torch.max(multiple_end-multiple_start)+1).item()
        range_offsets = torch.arange(max_range, device=device).unsqueeze(0)
        mutations = torch.clamp(multiple_start.unsqueeze(1)+range_offsets,max=seq_len-1)
        valid_mutations = mutations <= multiple_end.unsqueeze(1)
        batch_indices_multiple = batch_indices[multiple_mutation_mask].unsqueeze(1).expand(-1, max_range)
        multiple_diffs = fetch_pairwise_diff_multiple(mutations, masked_contact_repre, batch_indices_multiple, 
                                             multiple_mutation_mask.sum(), max_range, seq_len, padding_mask) # here we only compute the pairwise difference for the batch with multiple mutations
        multiple_diffs[~valid_mutations] = 0.0
        multiple_mutation_diffs = multiple_diffs.max(dim=1)[0]
    else:
        multiple_mutation_diffs = torch.tensor([], device=device)

    # Combine results
    result = torch.zeros(batch_size, num_mutations, device=device)
    result[single_mutation_mask] = single_mutation_diffs[single_mutation_mask]
    result[multiple_mutation_mask] = multiple_mutation_diffs
    return result
            
def fetch_global_contact(contact_repre, padding_mask):
    batch_size, seq_len = contact_repre.size(0), contact_repre.size(1)
    device = contact_repre.device

    # Expand padding mask
    padding_expand = (~padding_mask).unsqueeze(1) & (~padding_mask).unsqueeze(2)
    # Apply padding mask to contact_repre
    masked_contact_repre = contact_repre * padding_expand

    # Create a tensor of all possible residue indices
    all_residues = torch.arange(seq_len, device=device).expand(batch_size, seq_len)

    # Compute pairwise differences for all positions at once
    pairwise_residues = masked_contact_repre[torch.arange(batch_size, device=device)[:, None, None], all_residues[:, :, None], all_residues[:, None, :]]
    residue_vals = pairwise_residues.gather(2, all_residues.unsqueeze(2))
    pairwise_diff = ((pairwise_residues - residue_vals) ** 2 * (padding_expand)).sum(dim=2) / (seq_len-1)

    return pairwise_diff


if __name__ == '__main__':
    
    padding_mask = torch.tensor([[3,2,1,0],[2,3,1,0]])
    padding_mask = padding_mask.eq(0)
    contact_repre = torch.tensor([[[1,2,3,4],
                                   [2,3,4,5],
                                   [3,4,5,6],
                                   [4,5,6,7]],
                                  [[5,6,7,8],
                                   [6,7,8,9],
                                   [7,8,9,10],
                                   [8,9,10,11]]])
    seq_mutations = torch.tensor([[[0,0],[1,3]],
                                  [[0,2],[3,3]]])
    result = fetch_local_contact(contact_repre,seq_mutations,padding_mask)
    print('result: ',result)
