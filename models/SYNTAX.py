
import numpy as np
from support.modules import ConditionalRandomField, allowed_transitions
from modules.transformer import TransformerEncoder, AdaptedTransformerEncoder

from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
import math
from typing import Callable


class Highway(nn.Module):
    def __init__(self, size, num_layers, f):
        super(Highway, self).__init__()
        self.num_layers = num_layers
        self.nonlinear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.linear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.gate = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.f = f

    def forward(self, x):
        for layer in range(self.num_layers):
            gate = torch.sigmoid(self.gate[layer](x))
            nonlinear = self.f(self.nonlinear[layer](x))
            linear = self.linear[layer](x)
            x = gate * nonlinear + (1 - gate) * linear
        return x


class KeyValueMemoryNetwork(nn.Module):
    def __init__(self, vocab_size, feature_vocab_size, emb_size, scaled=False, temper=1, attn_type="dot", use_key=True,
                 key_embed_dropout=0.2, knowledge_type="all"):
        super(KeyValueMemoryNetwork, self).__init__()
        self.use_key = use_key
        if self.use_key:
            self.key_pos_embedding = nn.Embedding(vocab_size, emb_size, padding_idx=0)
            self.key_dep_embedding = nn.Embedding(vocab_size, emb_size, padding_idx=0)
            self.key_chunk_embedding = nn.Embedding(vocab_size, emb_size, padding_idx=0)
            self.key_embed_dropout = nn.Dropout(p=key_embed_dropout)

        self.knowledge_type = knowledge_type
        self.value_embedding = nn.Embedding(feature_vocab_size, emb_size, padding_idx=0)
        self.scaled = scaled
        self.scale = np.power(emb_size, 0.5 * temper)
        self.linear_pos = nn.Linear(2 * emb_size, 1)
        self.linear_dep = nn.Linear(2 * emb_size, 1)
        self.linear_chunk = nn.Linear(2 * emb_size, 1)
        self.softmax = nn.Softmax(dim=2)
        self.attn_type = attn_type
        if attn_type == "bilinear":
            self.weight = nn.Parameter(torch.Tensor(emb_size, emb_size))

    def forward(self, key_seq, pos_value_seq, dep_value_seq, chunk_value_seq, hidden,
                pos_mask_matrix, dep_mask_matrix, chunk_mask_matrix, nan_matrix):
        batch, seq_len, dim = hidden.shape
        pos_value_embed = self.value_embedding(pos_value_seq)
        dep_value_embed = self.value_embedding(dep_value_seq)
        chunk_value_embed = self.value_embedding(chunk_value_seq)
        key_pos_embed = self.key_pos_embedding(key_seq)
        key_dep_embed = self.key_dep_embedding(key_seq)
        key_chunk_embed = self.key_chunk_embedding(key_seq)
        key_pos_embed = self.key_embed_dropout(key_pos_embed)
        key_dep_embed = self.key_embed_dropout(key_dep_embed)
        key_chunk_embed = self.key_embed_dropout(key_chunk_embed)
        pos_value_embed = self.key_embed_dropout(pos_value_embed)
        dep_value_embed = self.key_embed_dropout(dep_value_embed)
        chunk_value_embed = self.key_embed_dropout(chunk_value_embed)
        if self.attn_type == "dot":
            u_pos = torch.bmm(hidden, key_pos_embed.transpose(1, 2))
            u_dep = torch.bmm(hidden, key_dep_embed.transpose(1, 2))
            u_chunk = torch.bmm(hidden, key_chunk_embed.transpose(1, 2))
        elif self.attn_type == "bilinear":
            u_pos = torch.bmm(hidden.matmul(self.weight), key_pos_embed.transpose(1, 2))
            u_dep = torch.bmm(hidden.matmul(self.weight), key_dep_embed.transpose(1, 2))
            u_chunk = torch.bmm(hidden.matmul(self.weight), key_chunk_embed.transpose(1, 2))
        if self.scaled:
            u_pos = u_pos / self.scale
            u_dep = u_dep / self.scale
            u_chunk = u_chunk / self.scale
        pos_mask_matrix = torch.clamp(pos_mask_matrix.float(), 0, 1)
        exp_u_pos = torch.exp(u_pos)
        delta_exp_u_pos = torch.mul(exp_u_pos, pos_mask_matrix)
        sum_delta_exp_u_pos = torch.stack([torch.sum(delta_exp_u_pos, 2)] * delta_exp_u_pos.shape[2], 2)
        p_pos = torch.div(delta_exp_u_pos, sum_delta_exp_u_pos + 1e-10)
        dep_mask_matrix = torch.clamp(dep_mask_matrix.float(), 0, 1)
        exp_u_dep = torch.exp(u_dep)
        delta_exp_u_dep = torch.mul(exp_u_dep, dep_mask_matrix)
        sum_delta_exp_u_dep = torch.stack([torch.sum(delta_exp_u_dep, 2)] * delta_exp_u_dep.shape[2], 2)
        p_dep = torch.div(delta_exp_u_dep, sum_delta_exp_u_dep + 1e-10)
        chunk_mask_matrix = torch.clamp(chunk_mask_matrix.float(), 0, 1)
        exp_u_chunk = torch.exp(u_chunk)
        delta_exp_u_chunk = torch.mul(exp_u_chunk, chunk_mask_matrix)
        sum_delta_exp_u_chunk = torch.stack([torch.sum(delta_exp_u_chunk, 2)] * delta_exp_u_chunk.shape[2], 2)
        p_chunk = torch.div(delta_exp_u_chunk, sum_delta_exp_u_chunk + 1e-10)
        o_pos = torch.bmm(p_pos, pos_value_embed)
        o_dep = torch.bmm(p_dep, dep_value_embed)
        o_chunk = torch.bmm(p_chunk, chunk_value_embed)
        o = torch.cat([o_pos.unsqueeze(2), o_dep.unsqueeze(2), o_chunk.unsqueeze(2)], dim=2)
        o = o.view(batch * seq_len, 3, -1)
        weight_pos = torch.sigmoid(self.linear_pos(torch.cat([hidden, o_pos], dim=2)))
        weight_dep = torch.sigmoid(self.linear_dep(torch.cat([hidden, o_dep], dim=2)))
        weight_chunk = torch.sigmoid(self.linear_chunk(torch.cat([hidden, o_chunk], dim=2)))
        weight = torch.cat([weight_pos, weight_dep, weight_chunk], dim=2)
        weight_binary = torch.softmax(weight, dim=2)
        weight_binary = weight_binary.view(batch * seq_len, 1, 3)
        o = torch.bmm(weight_binary, o).squeeze(1)
        o = o.view(batch, seq_len, -1)
        return o


class GateConcMechanism(nn.Module):
    def __init__(self, hidden_size=None):
        super(GateConcMechanism, self).__init__()
        self.hidden_size = hidden_size
        self.w1 = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.w2 = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.bias = nn.Parameter(torch.Tensor(self.hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv1 = 1. / math.sqrt(self.w1.size(1))
        stdv2 = 1. / math.sqrt(self.w2.size(1))
        stdv = (stdv1 + stdv2) / 2.
        self.w1.data.uniform_(-stdv1, stdv1)
        self.w2.data.uniform_(-stdv2, stdv2)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, hidden):
        gated = input.matmul(self.w1.t()) + hidden.matmul(self.w2.t()) + self.bias
        gate = torch.sigmoid(gated)
        output = torch.cat([input.mul(gate), hidden.mul(1 - gate)], dim=2)
        return output


class GateAddMechanism(nn.Module):
    def __init__(self, hidden_size=None):
        super(GateAddMechanism, self).__init__()
        self.hidden_size = hidden_size
        self.w1 = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.w2 = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.bias = nn.Parameter(torch.Tensor(self.hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv1 = 1. / math.sqrt(self.w1.size(1))
        stdv2 = 1. / math.sqrt(self.w2.size(1))
        stdv = (stdv1 + stdv2) / 2.
        self.w1.data.uniform_(-stdv1, stdv1)
        self.w2.data.uniform_(-stdv2, stdv2)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, hidden):
        gated = input.matmul(self.w1.t()) + hidden.matmul(self.w2.t()) + self.bias
        gate = torch.sigmoid(gated)
        output = input.mul(gate) + hidden.mul(1 - gate)
        return output


class LinearGateAddMechanism(nn.Module):
    def __init__(self, hidden_size=None):
        super(LinearGateAddMechanism, self).__init__()
        self.hidden_size = hidden_size
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.w1 = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.w2 = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.bias = nn.Parameter(torch.Tensor(self.hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv1 = 1. / math.sqrt(self.w1.size(1))
        stdv2 = 1. / math.sqrt(self.w2.size(1))
        stdv = (stdv1 + stdv2) / 2.
        self.w1.data.uniform_(-stdv1, stdv1)
        self.w2.data.uniform_(-stdv2, stdv2)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, hidden):
        input = self.linear1(input)
        hidden = self.linear2(hidden)
        gated = input.matmul(self.w1.t()) + hidden.matmul(self.w2.t()) + self.bias
        gate = torch.sigmoid(gated)
        output = input.mul(gate) + hidden.mul(1 - gate)
        return output


style_map = {
    'add': lambda x, y: x + y,
    'concat': lambda *args: torch.cat(args, args[0].dim() - 1),
    'diff': lambda x, y: x - y,
    'abs-diff': lambda x, y: torch.abs(x - y),
    'concat-diff': lambda x, y: torch.cat((x, y, x - y), x.dim() - 1),
    'concat-add': lambda x, y: torch.cat((x, y, x + y), x.dim() - 1),
    'concat-abs-diff': lambda x, y: torch.cat((x, y, torch.abs(x - y)), x.dim() - 1),
    'mul': lambda x, y: torch.mul(x, y),
    'concat-mul-diff': lambda x, y: torch.cat((x, y, torch.mul(x, y), torch.abs(x - y)), x.dim() - 1)
}


class FusionModule(nn.Module):
    def __init__(self, layer=1, fusion_type="concat", input_size=1024, output_size=1024, dropout=0.2):
        super(FusionModule, self).__init__()
        self.fusion_type = fusion_type
        self.layer = layer
        if self.layer > 0:
            self.highway = Highway(size=output_size, num_layers=layer, f=torch.nn.functional.relu)
        if self.fusion_type == "gate-add":
            self.gate = GateAddMechanism(hidden_size=input_size)
        elif self.fusion_type == "gate-concat":
            self.gate = GateConcMechanism(hidden_size=input_size)
        elif self.fusion_type == "l-gate-add":
            self.gate = LinearGateAddMechanism(hidden_size=input_size)
        self.fusion_dropout = nn.Dropout(p=dropout)

    def forward(self, enc_out, kv_out):
        if self.fusion_type in ["gate-add", "gate-concat", "l-gate-add"]:
            fused = self.gate(enc_out, kv_out)
        else:
            fused = style_map[self.fusion_type](enc_out, kv_out)
        fused = self.fusion_dropout(fused)
        if self.layer > 0:
            fused = self.highway(fused)
        return fused


_dim_map = {
    "concat": 2,
    "diff": 1,
    "gate-concat": 2,
    "gate-add": 1,
    "mul": 1,
    "abs-diff": 1,
    "concat-diff": 3,
    "concat-abs-diff": 3,
    "concat-mul-diff": 4,
    "add": 1,
    "concat-add": 3,
    'l-gate-add': 1
}


class SyntaxNER(nn.Module):
    def __init__(self, tag_vocab, embed, num_layers, d_model, n_head, feedforward_dim, dropout,
                 after_norm=True, attn_type='adatrans',  bi_embed=None,
                 fc_dropout=0.3, pos_embed=None, scale=False, dropout_attn=None,
                 use_knowledge=False,
                 feature2count=None,
                 vocab_size=None,
                 feature_vocab_size=None,
                 kv_attn_type="dot",
                 memory_dropout=0.2,
                 fusion_dropout=0.2,
                 fusion_type='concat',
                 highway_layer=0,
                 key_embed_dropout=0.2,
                 knowledge_type="all",
                 ):
        super().__init__()
        self.use_knowledge = use_knowledge
        self.feature2count = feature2count
        self.vocab_size = vocab_size
        self.feature_vocab_size = feature_vocab_size
        self.embed = embed
        embed_size = self.embed.embed_size
        self.bi_embed = None
        if bi_embed is not None:
            self.bi_embed = bi_embed
            embed_size += self.bi_embed.embed_size
        self.in_fc = nn.Linear(embed_size, d_model)
        self.transformer = TransformerEncoder(num_layers, d_model, n_head, feedforward_dim, dropout,
                                              after_norm=after_norm, attn_type=attn_type,
                                              scale=scale, dropout_attn=dropout_attn,
                                              pos_embed=pos_embed)
        self.kv_memory = KeyValueMemoryNetwork(vocab_size=vocab_size, feature_vocab_size=feature_vocab_size,
                                               attn_type=kv_attn_type, emb_size=d_model, scaled=True,
                                               key_embed_dropout=key_embed_dropout,
                                               knowledge_type=knowledge_type)
        self.output_dim = d_model * _dim_map[fusion_type]
        self.fusion = FusionModule(fusion_type=fusion_type, layer=highway_layer, input_size=d_model,
                                   output_size=self.output_dim, dropout=fusion_dropout)
        self.memory_dropout = nn.Dropout(p=memory_dropout)
        self.out_fc = nn.Linear(self.output_dim, len(tag_vocab))
        self.fc_dropout = nn.Dropout(fc_dropout)
        trans = allowed_transitions(tag_vocab, include_start_end=True)
        self.crf = ConditionalRandomField(len(tag_vocab), include_start_end_trans=True, allowed_transitions=trans)

    def _forward(self, chars, target, bigrams=None, pos_features=None, dep_features=None, chunk_features=None,
                 pos_matrix=None, dep_matrix=None, chunk_matrix=None, nan_matrix=None):
        mask = chars.ne(0)
        hidden = self.embed(chars)
        if self.bi_embed is not None:
            bigrams = self.bi_embed(bigrams)
            hidden = torch.cat([hidden, bigrams], dim=-1)
        hidden = self.in_fc(hidden)
        encoder_output = self.transformer(hidden, mask)
        kv_output = self.kv_memory(chars, pos_features, dep_features, chunk_features, hidden,
                                   pos_matrix, dep_matrix, chunk_matrix, nan_matrix)
        kv_output = self.memory_dropout(kv_output)
        concat = self.fusion(encoder_output, kv_output)
        concat = self.fc_dropout(concat)
        concat = self.out_fc(concat)
        logits = F.log_softmax(concat, dim=-1)
        if target is None:
            paths, _ = self.crf.viterbi_decode(logits, mask)
            return {'pred': paths}
        else:
            loss = self.crf(logits, target, mask)
            return {'loss': loss}

    def forward(self, chars, target, bigrams=None, pos_features=None, dep_features=None, chunk_features=None,
                pos_matrix=None, dep_matrix=None, chunk_matrix=None, nan_matrix=None):
        return self._forward(chars, target, bigrams, pos_features, dep_features, chunk_features,
                             pos_matrix, dep_matrix, chunk_matrix, nan_matrix)

    def predict(self, chars, bigrams=None, pos_features=None, dep_features=None, chunk_features=None,
                pos_matrix=None, dep_matrix=None, chunk_matrix=None, nan_matrix=None):
        return self._forward(chars, target=None, bigrams=bigrams,
                             pos_features=pos_features, dep_features=dep_features, chunk_features=chunk_features,
                             pos_matrix=pos_matrix, dep_matrix=dep_matrix, chunk_matrix=chunk_matrix,
                             nan_matrix=nan_matrix)
