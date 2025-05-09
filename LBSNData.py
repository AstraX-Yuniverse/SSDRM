import copy
import math
import time
import numpy as np
import joblib
import datetime
import gc
from tqdm import tqdm
from nltk import ngrams
from collections import defaultdict
from torch.utils.data import Dataset
from utils import build_st_matrix, extract_sub_matrix
from quadkey_encoder import latlng2quadkey
from torchtext.data import Field


def time_preprocessing(x):
    return [float(y) for y in x.split()]


class LBSNData(Dataset):
    def __init__(self, data_name, data_path, min_loc_freq, min_user_freq, map_level):
        self.loc2idx = {'<pad>': 0}
        self.loc2gps = {'<pad>': (0.0, 0.0)}
        self.idx2loc = {0: '<pad>'}
        self.idx2gps = {0: (0.0, 0.0)}
        self.loc2count = {}
        self.n_loc = 1
        self.n_timestamp = 1
        self.build_vocab(data_name, data_path, min_loc_freq)
        #print(self.n_loc)
        self.user_seq, self.user2idx, self.quadkey2idx, self.n_user, self.n_quadkey, self.quadkey2loc = \
            self.processing(data_name, data_path, min_user_freq, map_level)

    def build_vocab(self, data_name, data_path, min_loc_freq):
        for line in open(data_path, encoding='gbk'):
            line = line.strip().split('\t')
            if len(line) != 5:
                continue
            if data_name == 'weeplaces':
                # user, loc, time(2010-10-22T23:44:29), lat, lng
                loc = line[1]
                coordinate = float(line[3]), float(line[4])
            elif data_name == 'brightkite' or data_name == data_name:
                # user, time(2010-10-19T23:55:27Z), lat, lng, loc
                loc = line[4]
                coordinate = float(line[2]), float(line[3])
            elif data_name == 'cc':
                loc = line[2]
                coordinate = float(line[3]), float(line[4])
            self.add_location(loc, coordinate)
        if min_loc_freq > 0:
            self.n_loc = 1
            self.loc2idx = {'<pad>': 0}
            self.idx2loc = {0: '<pad>'}
            self.idx2gps = {0: (0.0, 0.0)}
            for loc in self.loc2count:
                if self.loc2count[loc] >= min_loc_freq:
                    self.add_location(loc, self.loc2gps[loc])
        self.locidx2freq = np.zeros(self.n_loc - 1, dtype=np.int32)
        for idx, loc in self.idx2loc.items():
            if idx != 0:
                self.locidx2freq[idx - 1] = self.loc2count[loc]

    def add_location(self, loc, coordinate):
        if loc not in self.loc2idx:
            self.loc2idx[loc] = self.n_loc
            self.loc2gps[loc] = coordinate
            self.idx2loc[self.n_loc] = loc
            self.idx2gps[self.n_loc] = coordinate
            if loc not in self.loc2count:
                self.loc2count[loc] = 1
            self.n_loc += 1
        else:
            self.loc2count[loc] += 1

    def processing(self, data_name, data_path, min_user_freq, map_level):
        print("Processing data...")
        user_seq = {}
        user_seq_array = list()
        quadkey2idx = {}
        idx2quadkey = {}
        quadidx2loc = defaultdict(set)
        n_quadkey = 1
        user2idx = {}
        n_user = 1

        for line in open(data_path, encoding='gbk'):
            line = line.strip().split('\t')
            if len(line) != 5:
                continue

            if data_name == 'weeplaces':
                # user, loc, time(2010-10-22T23:44:29), lat, lng
                user, loc, t, lat, lng = line
                lat = float(lat)
                lng = float(lng)
                t = time.strptime(t, '%Y-%m-%dT%H:%M:%S')
                timestamp = time.mktime(t)
            elif data_name == data_name or data_name == 'brightkite':
                # user, time(2010-10-19T23:55:27Z), lat, lng, loc
                user, t, lat, lng, loc = line
                lat = float(lat)
                lng = float(lng)
                t = time.strptime(t, '%Y-%m-%dT%H:%M:%SZ')
                timestamp = time.mktime(t)
            elif data_name == 'cc':
                # user, time(20101019235527), loc, lat, lng
                user, t, loc, lat, lng = line
                lat = float(lat)
                lng = float(lng)
                t = time.strptime(t, '%Y%m%d%H%M%S')
                timestamp = time.mktime(t)

            if loc not in self.loc2idx:
                continue

            loc_idx = self.loc2idx[loc]
            quadkey = latlng2quadkey(lat, lng, map_level)

            if quadkey not in quadkey2idx:
                quadkey2idx[quadkey] = n_quadkey
                idx2quadkey[n_quadkey] = quadkey
                n_quadkey += 1
            quadkey_idx = quadkey2idx[quadkey]
            quadidx2loc[quadkey_idx].add(loc_idx)

            if user not in user_seq:
                user_seq[user] = list()
            user_seq[user].append([loc_idx, quadkey, lat, lng, timestamp])

        print("Processing user sequences...")
        for user, seq in tqdm(user_seq.items()):
            if len(seq) >= min_user_freq:
                user2idx[user] = n_user
                user_idx = n_user
                seq_new = list()
                tmp_set = set()
                cnt = 0

                for loc, quadkey, lat, lng, timestamp in sorted(seq, key=lambda e: e[-1]):
                    dt = datetime.datetime.fromtimestamp(timestamp)
                    time_features = [
                        dt.month,
                        dt.day,
                        dt.weekday(),
                        dt.hour,
                        dt.minute // 15
                    ]

                    is_first = loc not in tmp_set
                    if is_first:
                        tmp_set.add(loc)
                        cnt += 1
                    seq_new.append((user_idx, loc, quadkey, timestamp, lat, lng, time_features, not is_first))

                if cnt > min_user_freq / 2:
                    n_user += 1
                    user_seq_array.append(seq_new)

        print("Processing quadkeys...")
        all_quadkeys = []
        for u in range(len(user_seq_array)):
            seq = user_seq_array[u]
            for i in range(len(seq)):
                check_in = seq[i]
                region_quadkey = check_in[2]
                region_quadkey_bigram = ' '.join([''.join(x) for x in ngrams(region_quadkey, 6)])
                region_quadkey_bigram = region_quadkey_bigram.split()
                all_quadkeys.append(region_quadkey_bigram)
                user_seq_array[u][i] = (
                    check_in[0],  # user_idx
                    check_in[1],  # loc
                    region_quadkey_bigram,  # quadkey
                    check_in[3],  # timestamp
                    check_in[4],  # lat
                    check_in[5],  # lng
                    check_in[6],  # time_features
                    check_in[7]  # is_first
                )

        self.loc2quadkey = ['NULL']
        for l in range(1, self.n_loc):
            lat, lng = self.idx2gps[l]
            quadkey = latlng2quadkey(lat, lng, map_level)
            quadkey_bigram = ' '.join([''.join(x) for x in ngrams(quadkey, 6)])
            self.loc2quadkey.append(quadkey_bigram.split())
            all_quadkeys.append(quadkey_bigram.split())

        self.GPSQUADKEY = Field(
            sequential=True,
            use_vocab=True,
            batch_first=True,
            unk_token=None,
            preprocessing=str.split
        )
        self.GPSQUADKEY.build_vocab(all_quadkeys)

        print("Encoding TIMECODES...")
        self.TIMECODE = Field(
            sequential=True,
            use_vocab=False,
            batch_first=True,
            preprocessing=time_preprocessing,
            include_lengths=False
        )

        return user_seq_array, user2idx, quadkey2idx, n_user, n_quadkey, quadidx2loc

    def __len__(self):
        return len(self.user_seq)

    def __getitem__(self, idx):
        return self.user_seq[idx]

    def data_partition(self, max_len):
        train_data = copy.copy(self)
        eval_data = copy.copy(self)
        train_seq = []
        eval_seq = []
        for user in tqdm(range(len(self)), desc="Partitioning data"):
            seq = self[user]
            for i in reversed(range(len(seq))):
                if not seq[i][-1]:
                    break
            eval_trg = seq[i: i + 1]
            eval_src_seq = seq[max(0, i - max_len): i]
            eval_seq.append((eval_src_seq, eval_trg))

            n_instance = math.floor((i + max_len - 1) / max_len)
            for k in range(n_instance):
                if (i - k * max_len) > max_len * 1.1:
                    train_trg_seq = seq[i - (k + 1) * max_len: i - k * max_len]
                    train_src_seq = seq[i - (k + 1) * max_len - 1: i - k * max_len - 1]
                    train_seq.append((train_src_seq, train_trg_seq))
                else:
                    train_trg_seq = seq[max(i - (k + 1) * max_len, 0): i - k * max_len]
                    train_src_seq = seq[max(i - (k + 1) * max_len - 1, 0): i - k * max_len - 1]
                    train_seq.append((train_src_seq, train_trg_seq))
                    break

        print("data_partition over")
        train_data.user_seq = train_seq
        eval_data.user_seq = eval_seq
        return train_data, eval_data

    def spatial_temporal_matrix_building(self, path):
        print("Building Spatial-Temporal Relation Matrix")
        user_matrix = {}
        for u in tqdm(range(len(self))):
            seq = self[u]
            temporal_matrix, spatial_matrix = build_st_matrix(self, seq, len(seq))
            user_matrix[u] = (temporal_matrix, spatial_matrix)
        joblib.dump(user_matrix, path)