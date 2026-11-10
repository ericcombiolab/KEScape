import math
import os
from collections import OrderedDict
import torch
import torch.nn as nn
import esm
import torch.nn.functional as F
import pretrained_new
from misc import fetch_local_embed

BACKBONE_CKPT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'checkpoint',
    'esmfold_finetuned_step4500_setting50.pt',
)


def maybe_patch_esm_sdpa(esm_model, enable=None):
    if enable is None:
        enable = os.environ.get('KESCAPE_ESM_SDPA', '0') == '1'
    if not enable:
        return False
    if not hasattr(F, 'scaled_dot_product_attention'):
        print('KESCAPE_ESM_SDPA=1 requested, but this torch (%s) has no '
              'F.scaled_dot_product_attention (needs >= 2.1); keeping the '
              'reference attention path.' % torch.__version__, flush=True)
        return False
    import types
    from esm.multihead_attention import MultiheadAttention as _MHA
    _orig_forward = _MHA.forward

    def _sdpa_forward(self, query, key, value, key_padding_mask=None,
                      incremental_state=None, need_weights=True, static_kv=False,
                      attn_mask=None, before_softmax=False, need_head_weights=False):
        if (need_head_weights or before_softmax
                or incremental_state is not None or static_kv
                or attn_mask is not None or self.bias_k is not None
                or self.add_zero_attn or not self.qkv_same_dim):
            return _orig_forward(self, query, key, value, key_padding_mask,
                                 incremental_state, need_weights, static_kv,
                                 attn_mask, before_softmax, need_head_weights)
        tgt_len, bsz, embed_dim = query.shape
        q = self.q_proj(query) * self.scaling
        k = self.k_proj(key)
        v = self.v_proj(value)
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if self.rot_emb is not None:
            q, k = self.rot_emb(q, k)
        q = q.reshape(bsz, self.num_heads, tgt_len, self.head_dim)
        k = k.reshape(bsz, self.num_heads, -1, self.head_dim)
        v = v.reshape(bsz, self.num_heads, -1, self.head_dim)
        am = None
        if key_padding_mask is not None and key_padding_mask.dim() > 0 and key_padding_mask.any():
            am = ~key_padding_mask.bool()[:, None, None, :]
        o = F.scaled_dot_product_attention(
            q, k, v, attn_mask=am,
            dropout_p=self.dropout if self.training else 0.0, scale=1.0)
        o = o.permute(2, 0, 1, 3).reshape(tgt_len, bsz, embed_dim)
        return self.out_proj(o), None

    n_patched = 0
    for mod in esm_model.esm.modules():
        if isinstance(mod, _MHA):
            mod.forward = types.MethodType(_sdpa_forward, mod)
            n_patched += 1
    print('KESCAPE_ESM_SDPA=1: patched %d ESM-2 attention modules to '
          'F.scaled_dot_product_attention' % n_patched, flush=True)
    return n_patched > 0

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

    def forward(self, seq_embed, structure_embed, seq_padding_mask=None, struc_padding_mask=None, need_attn=False):
        structure_embed = structure_embed.transpose(0, 1)

        seq_embed = self.proj_seq(seq_embed)
        residual = seq_embed
        residual = residual.transpose(0, 1)
        seq_embed = seq_embed.transpose(0, 1)

        if need_attn:
            attn1_weights, _ = self.attn1(seq_embed, structure_embed, structure_embed, key_padding_mask=struc_padding_mask, before_softmax=True,need_head_weights=True)
        else:
            attn1_weights = None
        fuse_embed, _ = self.attn1(seq_embed, structure_embed, structure_embed, key_padding_mask=struc_padding_mask, need_head_weights=True)


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
        if need_attn:
            attn2_weights, _ = self.attn2(fuse_embed, fuse_embed, fuse_embed, key_padding_mask=seq_padding_mask, before_softmax=True, need_head_weights=True)
        else:
            attn2_weights = None
        fuse_embed, _ = self.attn2(fuse_embed, fuse_embed, fuse_embed, key_padding_mask=seq_padding_mask, need_head_weights=True)

        fuse_embed = fuse_embed + residual
        residual = fuse_embed
        fuse_embed = self.layer_norm3(fuse_embed)
        fuse_embed = self.linear3(fuse_embed)
        fuse_embed = F.gelu(fuse_embed)
        fuse_embed = self.linear4(fuse_embed)
        fuse_embed = residual + fuse_embed
        fuse_embed = fuse_embed.transpose(0, 1)

        return fuse_embed, structure_embed, attn1_weights, attn2_weights


class RegressionLayer(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.dense1 = nn.Linear(embed_dim, embed_dim//2)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.dense1(x)

        x = F.gelu(x)
        x = self.dropout(x)
        embed = x.clone()

        return embed


class FitnessPredictor(nn.Module):
    def __init__(self,num_layers,num_heads,block_embed_dim):
        super(FitnessPredictor, self).__init__()

        self.num_layers = num_layers
        self.esm_model = pretrained_new._load_model(BACKBONE_CKPT_PATH)
        for _mod in self.esm_model.esm.modules():
            if isinstance(_mod, esm.multihead_attention.MultiheadAttention):
                _mod.enable_torch_version = False
        maybe_patch_esm_sdpa(self.esm_model)
        self.alphabet = esm.Alphabet.from_architecture('ESM-1b')
        self.alphabet_size = len(self.alphabet)
        self.block_embed_dim = block_embed_dim
        self.fitness_embed_dim = self.esm_model.esm_feats
        self.structure_embed_dim = 1024
        init_weight = torch.zeros(self.esm_model.esm.num_layers + 1)
        self.hidden_repre_combine = nn.Parameter(init_weight)

        for param in self.esm_model.parameters():
            param.requires_grad = False

        self._bb_cache = OrderedDict()
        self._bb_cache_cap = 64

        self.cross_block = CrossModalBlock(
            seq_dim = self.fitness_embed_dim,
            struc_dim = self.structure_embed_dim,
            num_heads = num_heads
        )

        self.regress_block = RegressionLayer(
            embed_dim = 3*self.structure_embed_dim,
        )

        self.dist_head = nn.Linear(1, 1)

        self.immune_escape_head = nn.Linear(3 * self.structure_embed_dim // 2, 1)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        for suffix in ('weight', 'bias'):
            old_key = prefix + 'aux_head.' + suffix
            new_key = prefix + 'immune_escape_head.' + suffix
            if old_key in state_dict:
                if new_key not in state_dict:
                    state_dict[new_key] = state_dict[old_key]
                del state_dict[old_key]
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def train(self, mode=True):
        super().train(mode)
        self.esm_model.eval()
        return self

    @torch.no_grad()
    def calibrate_dist_head(self, l2_mean, l2_std, label_mean, label_std,
                            slope_frac):
        l2_std = max(float(l2_std), 1e-6)
        s0 = slope_frac * float(label_std) / l2_std
        s0 = min(max(s0, 1e-4), 10.0)
        self.dist_head.weight.fill_(math.log(math.expm1(s0)))
        self.dist_head.bias.fill_(float(label_mean) - s0 * float(l2_mean))

    def _backbone_features(self, x, mask, cacheable=False):
        if not cacheable:
            with torch.no_grad():
                return self.esm_model.backbone_features(x, mask=mask)

        B = x.size(0)
        keys = []
        s_s_rows = [None] * B
        hr_rows = [None] * B
        miss = []
        for i in range(B):
            li = int(mask[i].sum().item())
            key = x[i, :li].cpu().numpy().tobytes()
            keys.append(li)
            hit = self._bb_cache.get(key)
            if hit is None:
                miss.append((i, key))
            else:
                self._bb_cache.move_to_end(key)
                s_s_rows[i], hr_rows[i] = hit
        if miss:
            miss_rows = [i for i, _ in miss]
            with torch.no_grad():
                s_s_m, hr_m = self.esm_model.backbone_features(
                    x[miss_rows], mask=mask[miss_rows])
            for j, (i, key) in enumerate(miss):
                li = keys[i]
                entry = (s_s_m[j, :li].clone(), hr_m[j, :li + 2].clone())
                self._bb_cache[key] = entry
                while len(self._bb_cache) > self._bb_cache_cap:
                    self._bb_cache.popitem(last=False)
                s_s_rows[i], hr_rows[i] = entry

        L = x.size(1)
        s_s = torch.zeros(B, L, s_s_rows[0].size(-1), device=x.device,
                          dtype=s_s_rows[0].dtype)
        hr = torch.zeros(B, L + 2, hr_rows[0].size(-2), hr_rows[0].size(-1),
                         device=x.device, dtype=hr_rows[0].dtype)
        for i in range(B):
            li = keys[i]
            s_s[i, :li] = s_s_rows[i]
            hr[i, :li + 2] = hr_rows[i]
        return s_s, hr

    def generate_embedding_virus_seq(self, x, mask, seq_mutations, cacheable=False, need_attn=False):
        s_s, hidden_repre = self._backbone_features(x, mask, cacheable=cacheable)
        structure_embed = s_s
        seq_embed = hidden_repre.to(self.hidden_repre_combine.dtype)
        seq_embed = (self.hidden_repre_combine.softmax(0).unsqueeze(0) @ seq_embed).squeeze(2)

        esm_seq_embed = seq_embed.clone()

        struc_padding_mask = ~mask.bool()
        bos_eos = torch.zeros(mask.size(0), 1, dtype=torch.bool, device=mask.device)
        seq_padding_mask = torch.cat([bos_eos, struc_padding_mask, bos_eos], dim=1)
        seq_embed, structure_embed, attn1_weights, attn2_weights = self.cross_block(
            seq_embed, structure_embed, seq_padding_mask, struc_padding_mask, need_attn=need_attn)

        seq_embed_local, _ = fetch_local_embed(seq_embed,structure_embed,seq_mutations)

        return seq_embed_local, attn1_weights, attn2_weights, seq_embed, esm_seq_embed

    def max_diff(self, K, raw_embed, mut_embed, seq1_mutations):
        batch_size, _, embed_dim = raw_embed.size()
        device = raw_embed.device

        avg_raw_embed = torch.zeros((batch_size, embed_dim), device=device, dtype=raw_embed.dtype)
        avg_mut_embed = torch.zeros((batch_size, embed_dim), device=device, dtype=mut_embed.dtype)

        for i in range(batch_size):
            valid_indices = torch.where(seq1_mutations[i, :, 0] != -1)[0]
            num_mutations = len(valid_indices)

            if num_mutations == 0:
                print(f"Warning: No valid mutations found for batch index {i}. Skipping this batch.")
                continue

            raw_i = raw_embed[i, valid_indices]
            mut_i = mut_embed[i, valid_indices]

            distances = torch.norm(mut_i - raw_i, p=2, dim=1)

            k = min(K, num_mutations)

            top_k_indices = torch.topk(distances, k).indices

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
        seq_embed1_local, attn1_weights1, attn2_weights1, seq_embed1, esm_seq_embed1 = self.generate_embedding_virus_seq(x1, mask1, seq1_mutations, cacheable=True, need_attn=True)
        seq_embed2_local, attn1_weights2, attn2_weights2, seq_embed2, esm_seq_embed2 = self.generate_embedding_virus_seq(x2, mask2, seq2_mutations, cacheable=False, need_attn=True)

        seq_embed1_local, seq_embed2_local = self.max_diff(5, seq_embed1_local, seq_embed2_local, seq1_mutations)
        output = torch.cat([seq_embed1_local,seq_embed2_local,seq_embed2_local-seq_embed1_local], dim=1)

        embed = self.regress_block(output)
        return embed, attn1_weights1, attn1_weights2, attn2_weights1, attn2_weights2, seq_embed1, seq_embed2, esm_seq_embed1, esm_seq_embed2

    def forward(self, x1, x2, mask1, mask2, seq1_mutations, seq2_mutations):
        seq_embed1_local, _, _, _, _ = self.generate_embedding_virus_seq(x1, mask1, seq1_mutations, cacheable=True)
        seq_embed2_local, _, _, _, _ = self.generate_embedding_virus_seq(x2, mask2, seq2_mutations, cacheable=False)

        avg_l2_per_sample = self.compute_avg_mutation_l2(seq_embed1_local, seq_embed2_local, seq1_mutations)

        seq_embed1_local, seq_embed2_local = self.max_diff(5, seq_embed1_local, seq_embed2_local, seq1_mutations)
        output = torch.cat([seq_embed1_local,seq_embed2_local,seq_embed2_local-seq_embed1_local], dim=1)

        embed = self.regress_block(output)
        dist_score = F.linear(avg_l2_per_sample.unsqueeze(-1),
                              F.softplus(self.dist_head.weight),
                              self.dist_head.bias).squeeze(-1)
        immune_escape_score = self.immune_escape_head(embed).squeeze(-1)
        return immune_escape_score, avg_l2_per_sample, dist_score
