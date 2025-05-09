import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
import math


def clones(module, num_sub_layer):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num_sub_layer)])


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_mark, d_model):
        super(TimeFeatureEmbedding, self).__init__()
        self.embed = nn.Linear(d_mark, d_model)

    def forward(self, x):
        return self.embed(x)


class Embedding(nn.Module):
    def __init__(self, vocab_size, num_units, zeros_pad=True, scale=False):
        super(Embedding, self).__init__()
        self.vocab_size = vocab_size
        self.num_units = num_units
        self.zeros_pad = zeros_pad
        self.scale = scale
        self.lookup_table = nn.Parameter(torch.Tensor(vocab_size, num_units))
        nn.init.xavier_normal_(self.lookup_table.data)
        if self.zeros_pad:
            self.lookup_table.data[0, :].fill_(0)

    def forward(self, inputs):
        if self.zeros_pad:
            self.padding_idx = 0
        else:
            self.padding_idx = -1
        outputs = F.embedding(inputs, self.lookup_table,
                              self.padding_idx, None, 2, False, False)
        if self.scale:
            outputs = outputs * (self.num_units ** 0.5)

        return outputs


class SubLayerConnect(nn.Module):
    def __init__(self, features):
        super(SubLayerConnect, self).__init__()
        self.norm = nn.LayerNorm(features)

    def forward(self, x, sublayer):
        # (*, d)
        return x + sublayer(self.norm(x))


class FFN(nn.Module):
    def __init__(self, features, exp_factor, dropout):
        super(FFN, self).__init__()
        self.w_1 = nn.Linear(features, exp_factor * features)
        self.act = nn.ReLU()
        self.w_2 = nn.Linear(exp_factor * features, features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w_1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.w_2(x)
        x = self.dropout(x)
        return x


class SA(nn.Module):
    def __init__(self, features, dropout):
        super(SA, self).__init__()
        # update q,k,v
        self.q_ = nn.Linear(features, features)
        self.k_ = nn.Linear(features, features)
        self.v_ = nn.Linear(features, features)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask, pad_mask):
        q = self.q_(x)
        k = self.k_(x)
        v = self.v_(x)
        scale_term = math.sqrt(q.size(-1))
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale_term
        if pad_mask is not None:
            scores = scores.masked_fill(pad_mask == 0.0, -1e9)
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            scores = scores.masked_fill(attn_mask == 0.0, -1e9)
        prob = F.softmax(scores, dim=-1)
        prob = self.dropout(prob)
        return torch.matmul(prob, v)


class GeoEncoderLayer(nn.Module):
    def __init__(self, features, exp_factor, dropout):
        super(GeoEncoderLayer, self).__init__()
        self.sa_layer = SA(features, dropout)
        self.ffn_layer = FFN(features, exp_factor, dropout)
        self.sublayer = clones(SubLayerConnect(features), 2)

    def forward(self, x):
        # (b ,n, l, d)
        x = self.sublayer[0](x, lambda x: self.sa_layer(x, None, None))
        x = self.sublayer[1](x, self.ffn_layer)
        return x


class GeoEncoder(nn.Module):
    def __init__(self, features, layer, depth):
        super(GeoEncoder, self).__init__()
        self.layers = clones(layer, depth)
        self.norm = nn.LayerNorm(features)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = x.mean(dim=-2)
        return self.norm(x)


class TSCAB(nn.Module):
    def __init__(self, features, dropout):
        super().__init__()
        self.q_ = nn.Linear(features, features)
        self.k_ = nn.Linear(features, features)
        self.v_ = nn.Linear(features * 2, features*2)
        self.norm = nn.LayerNorm(features*2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg_time, src_time,  trg_poi, attn_mask, pad_mask):
        q = self.q_(trg_time)
        k = self.k_(src_time)
        v = self.v_(trg_poi)
        scale_term = math.sqrt(q.size(-1))
        score = torch.matmul(q, k.transpose(-2, -1)) / scale_term
        if pad_mask is not None:
            score.masked_fill(pad_mask == 0.0, -1e9)
        if attn_mask is not None:
            attn_mask.unsqueeze(0)
            score.masked_fill(attn_mask == 0.0, -1e9)
        prob = F.softmax(score, dim=-1)
        prob = self.dropout(prob)
        return self.norm(torch.matmul(prob, v))


class TIAB(nn.Module):
    def __init__(self, features, exp_factor, dropout):
        super().__init__()
        self.q_ = nn.Linear(features, features)
        self.k_ = nn.Linear(features, features)
        self.v_ = nn.Linear(features, features)
        self.ffn_layer = FFN(features, exp_factor, dropout)
        self.norm = nn.LayerNorm(features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, user_emb, src_emb, trg_emb,t_mat, attn_mask, pad_mask):
        q = self.q_(user_emb)
        k = self.k_(src_emb)
        v = self.v_(trg_emb)
        scale_term = math.sqrt(q.size(-1))
        score = torch.matmul(q, k.transpose(-2, -1)) / scale_term
        t_mat = F.softmax(t_mat,-1)
        if pad_mask is not None:
            t_mat.masked_fill(pad_mask == 0.0, -1e9)
            score.masked_fill(pad_mask == 0.0, -1e9)
        if attn_mask is not None:
            attn_mask.unsqueeze(0)
            t_mat.masked_fill(attn_mask == 0.0, -1e9)
            score.masked_fill(attn_mask == 0.0, -1e9)
        prob = F.softmax(score, dim=-1) + t_mat
        prob = self.dropout(prob)
        prob = q + torch.matmul(prob, v)
        prob = prob + self.ffn_layer(prob)
        return self.norm(prob)
