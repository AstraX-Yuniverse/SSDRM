import torch
import torch.nn as nn
import torch.nn.functional as F
from time_aware_pe import TAPE
from attn_modules import *

class MultiHeadAttention(nn.Module):
    def __init__(self, features, num_heads, dropout):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = features // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_linear = nn.Linear(features, features)
        self.k_linear = nn.Linear(features, features)
        self.v_linear = nn.Linear(features, features)
        self.dropout = nn.Dropout(dropout)
        self.out_linear = nn.Linear(features, features)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        # Linear projections
        q = self.q_linear(q).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(k).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(v).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled Dot-Product Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        output = torch.matmul(attn, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        output = self.out_linear(output)

        return output

class STiSAN(nn.Module):
    def __init__(self, n_user, n_loc, n_quadkey, n_timestamp, features, exp_factor, k_t, k_d, depth, src_len, dropout, device):
        super(STiSAN, self).__init__()
        self.src_len = src_len
        self.device = device
        self.emb_loc = Embedding(n_loc, features, True, True)
        self.emb_quadkey = Embedding(n_quadkey, features, True, True)

        # 增加 GeoEncoder 的深度
        self.geo_encoder_layer = GeoEncoderLayer(features, exp_factor, dropout)
        self.geo_encoder = GeoEncoder(features, self.geo_encoder_layer, depth=4)  # 增加到 4 层

        self.timefeature = TimeFeatureEmbedding(5, features)
        self.k_t = torch.tensor(k_t)
        self.k_d = torch.tensor(k_d)  # Distance factor weight

        # 使用多头注意力机制
        self.multi_head_attn = MultiHeadAttention(features, num_heads=8, dropout=dropout)

        self.tscab = TSCAB(features, dropout)
        self.tiab = TIAB(features * 2, exp_factor, dropout)

    def calc_distance_score(self, src_lat, src_lng, src_size, trg_size):
        # src_lat, src_lng: [batch_size, max_len]
        # 只使用最后一个时间步的位置来计算距离
        last_valid_idx = (src_size - 1).to(self.device)  # [batch_size]
        batch_size = src_lat.size(0)

        # 获取每个序列的最后一个位置的经纬度
        batch_indices = torch.arange(batch_size, device=self.device)
        last_lat = src_lat[batch_indices, last_valid_idx]  # [batch_size]
        last_lng = src_lng[batch_indices, last_valid_idx]  # [batch_size]

        # 将经纬度转换为弧度
        last_lat = last_lat * torch.pi / 180.0
        last_lng = last_lng * torch.pi / 180.0

        # 扩展维度以便广播
        last_lat = last_lat.unsqueeze(1)  # [batch_size, 1]
        last_lng = last_lng.unsqueeze(1)  # [batch_size, 1]

        # 复制以匹配目标序列长度
        last_lat = last_lat.repeat(1, trg_size)  # [batch_size, trg_size]
        last_lng = last_lng.repeat(1, trg_size)  # [batch_size, trg_size]

        # 假设目标位置就是源位置（因为我们没有真实的目标位置）
        trg_lat = last_lat  # [batch_size, trg_size]
        trg_lng = last_lng  # [batch_size, trg_size]

        # Haversine公式计算距离
        dlat = trg_lat - last_lat
        dlng = trg_lng - last_lng

        a = torch.sin(dlat / 2) ** 2 + torch.cos(last_lat) * torch.cos(trg_lat) * torch.sin(dlng / 2) ** 2
        c = 2 * torch.asin(torch.sqrt(torch.clamp(a, min=0.0, max=1.0)))
        r = 6371  # 地球半径（公里）

        distances = c * r

        # 将距离转换为分数（距离越近分数越高）
        distance_scores = torch.exp(-distances / self.k_d)

        return distance_scores  # [batch_size, trg_size]

    def forward(self, src_user, src_loc, src_quadkey, src_time, src_timecodes, src_lat, src_lon, pad_mask, attn_mask,
                trg_loc, trg_quadkey, trg_time, trg_time_grams, key_pad_mask, mem_mask, ds, training):

        src_timecodes = src_timecodes.float()
        trg_time_grams = trg_time_grams.float()

        # Original embeddings and features
        src_loc_emb = self.emb_loc(src_loc)
        src_quadkey_emb = self.emb_quadkey(src_quadkey)
        src_timefeature = self.timefeature(src_timecodes)
        src_quadkey_emb = self.geo_encoder(src_quadkey_emb)
        trg_quadkey_emb = self.emb_quadkey(trg_quadkey)
        trg_timefeature = self.timefeature(trg_time_grams)

        # Concatenate location and quadkey embeddings
        src = torch.cat([src_loc_emb, src_quadkey_emb], dim=-1)

        # Get temporal matrix
        if training:
            trg_tmat = self.get_tmat_train(src_time, trg_time, self.k_t)
        else:
            trg_tmat = self.get_tmat_eval(src_time, trg_time, self.k_t)

        # Process through attention layers
        src_person = self.tscab(trg_timefeature, src_timefeature, src, attn_mask, pad_mask)
        if training:
            src = self.tiab(src_person, src, src, trg_tmat, attn_mask, pad_mask)
        else:
            src = self.tiab(src_person, src, src, trg_tmat, None, None)

        # Get target embeddings
        trg_loc_emb = self.emb_loc(trg_loc)
        trg_quadkey_emb = self.geo_encoder(trg_quadkey_emb)
        trg = torch.cat([trg_loc_emb, trg_quadkey_emb], dim=-1)

        # Repeat source embeddings for each target
        src = src.repeat(1, trg.size(1) // src.size(1), 1)

        # Calculate embedding similarity score
        sim_score = torch.sum(src * trg, dim=-1)  # [batch_size, trg_size]

        # Calculate distance score
        dist_score = self.calc_distance_score(src_lat, src_lon, torch.tensor(ds), trg.size(1))

        # Combine scores (now they have the same shape)
        output = sim_score + dist_score

        return output

    def get_tmat_train(self, src_time, trg_time, k_t):
        max_len = self.src_len
        time_mat_i = trg_time.unsqueeze(-1).expand([-1, max_len, max_len]).to(self.device)
        time_mat_j = src_time.unsqueeze(1).expand([-1, max_len, max_len]).to(self.device)
        time_mat = torch.abs(time_mat_i - time_mat_j) / (3600. * 24)
        time_mat_max = (torch.ones_like(time_mat) * k_t)
        time_mat_ = torch.where(time_mat > time_mat_max, time_mat_max, time_mat) - time_mat
        return time_mat_

    def get_tmat_eval(self, src_time, trg_time, k_t):
        max_len = self.src_len
        time_mat_i = src_time
        time_mat_j = trg_time.expand([-1, max_len]).to(self.device)
        time_mat = torch.abs(time_mat_i - time_mat_j) / (3600. * 24)
        time_mat_max = (torch.ones_like(time_mat) * k_t)
        time_mat_ = torch.where(time_mat > time_mat_max, time_mat_max, time_mat) - time_mat
        return time_mat_.unsqueeze(1)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))