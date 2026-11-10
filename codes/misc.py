import torch


def fetch_local_embed(seq_embed, structure_embed, seq_mutations):
    max_mut_num = 0
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
                seq_embed_local[batch, i] = sub_seq_embed[start_index+1]
                structure_embed_local[batch, i] = sub_structure_embed[start_index]
            else:
                seq_embed_local[batch, i] = sub_seq_embed[start_index+1:end_index+2].mean(dim=0)
                structure_embed_local[batch, i] = sub_structure_embed[start_index:end_index+1].mean(dim=0)
    if torch.all(seq_embed_local.eq(0)) or torch.all(structure_embed_local.eq(0)):
        raise Exception('Error: No mutation found in the sequence')
    return seq_embed_local, structure_embed_local
