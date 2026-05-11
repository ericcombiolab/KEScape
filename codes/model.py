import torch
import torch.nn as nn
import pandas as pandas
import esm
import math
import torch.nn.functional as F
import pretrained_new
from misc import fetch_local_embed

class CrossModalBlock(nn.Module):
    def __init__(self, seq_dim, struc_dim, num_heads):
        super(CrossModalBlock, self).__init__()
        self.num_heads = num_heads
        self.struc_dim = struc_dim
        self.seq_dim = seq_dim
        self.attn1 = esm.multihead_attention.MultiheadAttention(
            struc_dim,
            num_heads,
            add_zero_attn=False,
            use_rotary_embeddings=False,
        )
        self.attn2 = esm.multihead_attention.MultiheadAttention(
            struc_dim,
            num_heads,
            add_zero_attn=False,
            use_rotary_embeddings=False,
        )
        self.proj_seq = nn.Linear(seq_dim, struc_dim)
        self.linear1 = nn.Linear(struc_dim, 4*struc_dim)
        self.linear2 = nn.Linear(4*struc_dim, struc_dim)
        self.linear3 = nn.Linear(struc_dim, 4*struc_dim)
        self.linear4 = nn.Linear(4*struc_dim, struc_dim)
        self.layer_norm1 = nn.LayerNorm(struc_dim)
        self.layer_norm2 = nn.LayerNorm(struc_dim)
        self.layer_norm3 = nn.LayerNorm(struc_dim)
        
    
    def forward(self, seq_embed, structure_embed):
        structure_embed = structure_embed.transpose(0, 1)  # (batch_size, seq_len, struc_dim) -> (seq_len, batch_size, struc_dim)
        
        seq_embed = self.proj_seq(seq_embed) # (batch_size, seq_len+2, seq_dim) -> (batch_size, seq_len+2, struc_dim)
        residual = seq_embed
        residual = residual.transpose(0, 1) 
        seq_embed = seq_embed.transpose(0, 1)  # (batch_size, seq_len, seq_dim) -> (seq_len, batch_size, seq_dim)
        
        attn1_weights, _ = self.attn1(seq_embed, structure_embed, structure_embed, before_softmax=True,need_head_weights=True)
        fuse_embed, _ = self.attn1(seq_embed, structure_embed, structure_embed, need_head_weights=True)
        
        
        structure_embed = structure_embed.transpose(0, 1)
        
        fuse_embed = fuse_embed + residual
        residual = fuse_embed
        
        fuse_embed = self.layer_norm1(fuse_embed)
        fuse_embed = self.linear1(fuse_embed)
        fuse_embed = F.gelu(fuse_embed)
        fuse_embed = self.linear2(fuse_embed)
        fuse_embed = residual + fuse_embed
        residual = fuse_embed
        
        fuse_embed = self.layer_norm2(fuse_embed)
        attn2_weights, _ = self.attn2(fuse_embed, fuse_embed, fuse_embed, before_softmax=True, need_head_weights=True)
        fuse_embed, _ = self.attn2(fuse_embed, fuse_embed, fuse_embed, need_head_weights=True)
        
        fuse_embed = fuse_embed + residual
        residual = fuse_embed
        fuse_embed = self.layer_norm3(fuse_embed)
        fuse_embed = self.linear3(fuse_embed)
        fuse_embed = F.gelu(fuse_embed)
        fuse_embed = self.linear4(fuse_embed)
        fuse_embed = residual + fuse_embed
        fuse_embed = fuse_embed.transpose(0, 1)
        
        return fuse_embed, structure_embed, attn1_weights, attn2_weights
        

class ClassificationLayer(nn.Module):
    def __init__(self, embed_dim, output_dim):
        super().__init__()
        self.dense1 = nn.Linear(embed_dim, embed_dim//2)
        self.dense2 = nn.Linear(embed_dim//2, output_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.dense1(x)
        
        x = F.gelu(x)
        x = self.dropout(x)
        embed = x.clone()  # Save the intermediate embedding for later use
        x = self.dense2(x)
        x_soft = self.softmax(x)

        return x, x_soft, embed
        

class FitnessPredictor(nn.Module):
    def __init__(self,num_layers,num_heads,block_embed_dim,output_dim=2):
        super(FitnessPredictor, self).__init__()
        
        self.num_layers = num_layers
        self.esm_model = pretrained_new._load_model('./esmfold_finetuned.pt')
        self.alphabet = esm.Alphabet.from_architecture('ESM-1b')
        self.alphabet_size = len(self.alphabet)
        self.block_embed_dim = block_embed_dim
        self.output_dim = output_dim
        self.fitness_embed_dim = self.esm_model.esm_feats
        self.structure_embed_dim = 1024
        init_weight = torch.zeros(self.esm_model.esm.num_layers + 1)
        self.hidden_repre_combine = nn.Parameter(init_weight)
        
        # freeze the parameters of the ESM model
        for param in self.esm_model.parameters():
            param.requires_grad = False
        
        self.cross_block = CrossModalBlock(
            seq_dim = self.fitness_embed_dim,
            struc_dim = self.structure_embed_dim,
            num_heads = num_heads
        )
        
        # classification block
        self.classify_block = ClassificationLayer(
            embed_dim = 3*self.structure_embed_dim,
            output_dim = self.output_dim
        )
    
    def generate_embedding_virus_seq(self, x, mask, seq_mutations):
        padding_mask1 = mask.eq(0)
        padding_mask2 = mask.eq(0)
        first_true = (padding_mask2.cumsum(dim=1) == 1).max(dim=1)
        padding_mask2[range(mask.shape[0]), first_true.indices] = False
        bos_tensor = padding_mask2.new_full((padding_mask2.size(0), 1), False)
        eos_tensor = first_true.values.view(-1,1).to(padding_mask2.device)
        padding_mask2 = torch.cat([bos_tensor, padding_mask2, eos_tensor], dim=1)
        padding_mask3 = torch.cat([~bos_tensor, padding_mask1, ~bos_tensor], dim=1)
        
        structure, seq_embed = self.esm_model(x,mask=mask)
        seq_embed = seq_embed.to(self.hidden_repre_combine.dtype) 
        seq_embed = (self.hidden_repre_combine.softmax(0).unsqueeze(0) @ seq_embed).squeeze(2) # size: (batch_size, seq_len+2, embed_dim)
        
        esm_seq_embed = seq_embed.clone() 
        structure_embed = structure['s_s'] # size: (batch_size, seq_len, embed_dim)
        
        seq_embed, structure_embed, attn1_weights, attn2_weights = self.cross_block(seq_embed,structure_embed)
        
        seq_embed_local, _ = fetch_local_embed(seq_embed,structure_embed,seq_mutations)
        
        return seq_embed_local, attn1_weights, attn2_weights, seq_embed, esm_seq_embed
    
    def max_diff(self, K, raw_embed, mut_embed, seq1_mutations):
        """
        Calculates the L2 distance between raw and mutated embeddings for each mutation.
        Selects the embeddings (raw, mutated, interaction1, interaction2) corresponding
        to the top K largest distances and averages them.
        
        Returns:
            A tuple containing the averaged embeddings:
            (avg_raw_embed, avg_mut_embed, avg_int1_embed, avg_int2_embed)
            Each tensor has shape (batch_size, embed_dim or interaction_dim).
        """
        batch_size, _, embed_dim = raw_embed.size()
        device = raw_embed.device

        # Initialize output tensors
        avg_raw_embed = torch.zeros((batch_size, embed_dim), device=device, dtype=raw_embed.dtype)
        avg_mut_embed = torch.zeros((batch_size, embed_dim), device=device, dtype=mut_embed.dtype)

        for i in range(batch_size):
            # Find indices j where a mutation exists based on seq1_mutations
            valid_indices = torch.where(seq1_mutations[i, :, 0] != -1)[0]
            num_mutations = len(valid_indices)

            if num_mutations == 0:
                print(f"Warning: No valid mutations found for batch index {i}. Skipping this batch.")
                continue

            # Select embeddings for valid mutation positions
            raw_i = raw_embed[i, valid_indices]
            mut_i = mut_embed[i, valid_indices]

            # Calculate L2 distances between raw and mutated embeddings at mutation sites
            distances = torch.norm(mut_i - raw_i, p=2, dim=1)

            # Determine number of top distances to consider (up to K and number of mutations)
            k = min(K, num_mutations)

            # Get indices of the top k largest distances
            top_k_indices = torch.topk(distances, k).indices

            # Average the corresponding embeddings using the top k indices
            avg_raw_embed[i] = torch.mean(raw_i[top_k_indices], dim=0)
            avg_mut_embed[i] = torch.mean(mut_i[top_k_indices], dim=0)

        return avg_raw_embed, avg_mut_embed
    
    def compute_avg_mutation_l2(self, raw_embed, mut_embed, seq1_mutations):
        batch_size = raw_embed.size(0)
        device = raw_embed.device
        avg_l2_per_sample = torch.zeros(batch_size, device=device, dtype=raw_embed.dtype)

        for i in range(batch_size):
            valid_indices = torch.where(seq1_mutations[i, :, 0] != -1)[0]
            if len(valid_indices) == 0:
                continue

            raw_i = raw_embed[i, valid_indices]
            mut_i = mut_embed[i, valid_indices]
            distances = torch.norm(mut_i - raw_i, p=2, dim=1)
            avg_l2_per_sample[i] = distances.mean()

        return avg_l2_per_sample
    
    def extract_embedding(self, x1, x2, mask1, mask2, seq1_mutations, seq2_mutations):
        seq_embed1_local, attn1_weights1, attn2_weights1, seq_embed1, esm_seq_embed1 = self.generate_embedding_virus_seq(x1, mask1, seq1_mutations)
        seq_embed2_local, attn1_weights2, attn2_weights2, seq_embed2, esm_seq_embed2 = self.generate_embedding_virus_seq(x2, mask2, seq2_mutations)
        
        seq_embed1_local, seq_embed2_local = self.max_diff(5, seq_embed1_local, seq_embed2_local, seq1_mutations)
        output = torch.cat([seq_embed1_local,seq_embed2_local,seq_embed2_local-seq_embed1_local], dim=1)
        
        _, _, embed = self.classify_block(output)
        return embed, attn1_weights1, attn1_weights2, attn2_weights1, attn2_weights2, seq_embed1, seq_embed2, esm_seq_embed1, esm_seq_embed2
    
    def forward(self, x1, x2, mask1, mask2, seq1_mutations, seq2_mutations):
        seq_embed1_local, _, _, _, _ = self.generate_embedding_virus_seq(x1, mask1, seq1_mutations)
        seq_embed2_local, _, _, _, _ = self.generate_embedding_virus_seq(x2, mask2, seq2_mutations)
        
        avg_l2_per_sample = self.compute_avg_mutation_l2(seq_embed1_local, seq_embed2_local, seq1_mutations)

        seq_embed1_local, seq_embed2_local = self.max_diff(5, seq_embed1_local, seq_embed2_local, seq1_mutations)
        output = torch.cat([seq_embed1_local,seq_embed2_local,seq_embed2_local-seq_embed1_local], dim=1)
        
        output, output_soft, _ = self.classify_block(output)
        return output_soft, output, avg_l2_per_sample



