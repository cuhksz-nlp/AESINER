import torch
from torch import nn
import torch.nn.functional as F
import math


class RelativeEmbedding(nn.Module):
    def forward(self, input):
        bsz, seq_len = input.size()
        max_pos = self.padding_idx + seq_len
        if max_pos > self.origin_shift:
            weights = self.get_embedding(
                max_pos*2,
                self.embedding_dim,
                self.padding_idx,
            )
            weights = weights.to(self._float_tensor)
            del self.weights
            self.origin_shift = weights.size(0)//2
            self.register_buffer('weights', weights)

        positions = torch.arange(-seq_len, seq_len).to(input.device).long() + self.origin_shift  # 2*seq_len
        embed = self.weights.index_select(0, positions.long()).detach()
        return embed


class RelativeSinusoidalPositionalEmbedding(RelativeEmbedding):
    def __init__(self, embedding_dim, padding_idx, init_size=1568):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        assert init_size%2==0
        weights = self.get_embedding(
            init_size+1,
            embedding_dim,
            padding_idx,
        )
        self.register_buffer('weights', weights)
        self.register_buffer('_float_tensor', torch.FloatTensor(1))

    def get_embedding(self, num_embeddings, embedding_dim, padding_idx=None):
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(-num_embeddings//2, num_embeddings//2, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        self.origin_shift = num_embeddings//2 + 1
        return emb


class RelativeMultiHeadAttn(nn.Module):
    def __init__(self, d_model, n_head, dropout, r_w_bias=None, r_r_bias=None, scale=False):
        super().__init__()
        self.qv_linear = nn.Linear(d_model, d_model * 2, bias=False)
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.dropout_layer = nn.Dropout(dropout)

        self.pos_embed = RelativeSinusoidalPositionalEmbedding(d_model//n_head, 0, 1200)

        if scale:
            self.scale = math.sqrt(d_model // n_head)
        else:
            self.scale = 1

        if r_r_bias is None or r_w_bias is None:
            self.r_r_bias = nn.Parameter(nn.init.xavier_normal_(torch.zeros(n_head, d_model // n_head)))
            self.r_w_bias = nn.Parameter(nn.init.xavier_normal_(torch.zeros(n_head, d_model // n_head)))
        else:
            self.r_r_bias = r_r_bias
            self.r_w_bias = r_w_bias

    def forward(self, x, mask):
        batch_size, max_len, d_model = x.size()
        pos_embed = self.pos_embed(mask)

        qv = self.qv_linear(x)
        q, v = torch.chunk(qv, chunks=2, dim=-1)
        q = q.view(batch_size, max_len, self.n_head, -1).transpose(1, 2)
        k = x.view(batch_size, max_len, self.n_head, -1).transpose(1, 2)
        v = v.view(batch_size, max_len, self.n_head, -1).transpose(1, 2)

        rw_head_q = q + self.r_r_bias[:, None]
        AC = torch.einsum('bnqd,bnkd->bnqk', [rw_head_q, k])

        D_ = torch.einsum('nd,ld->nl', self.r_w_bias, pos_embed)[None, :, None]
        B_ = torch.einsum('bnqd,ld->bnql', q, pos_embed)
        BD = B_ + D_
        BD = self._shift(BD)
        attn = AC + BD

        attn = attn / self.scale

        attn = attn.masked_fill(mask[:, None, None, :].eq(0), float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout_layer(attn)
        v = torch.matmul(attn, v).transpose(1, 2).reshape(batch_size, max_len, d_model)

        return v

    def _shift(self, BD):
        bsz, n_head, max_len, _ = BD.size()
        zero_pad = BD.new_zeros(bsz, n_head, max_len, 1)
        BD = torch.cat([BD, zero_pad], dim=-1).view(bsz, n_head, -1, max_len)
        BD = BD[:, :, :-1].view(bsz, n_head, max_len, -1)
        BD = BD[:, :, :, max_len:]
        return BD
