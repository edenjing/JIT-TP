#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''SSE Model'''
import math
import os
import datetime as dt
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import json


class FeatureGenerator(object):
    def __init__(self, workspace, region_num):
        # self.csm = SparseDAM(workspace, seg_num, mask_size)
        self.workspace = workspace
        self.region_size = region_num
        self.PAD_ID = 0
        # self.time_delta = time_delta
        # self.num_1h = int(60 * 60 / self.time_delta)
        # self.num_1d = 24 * self.num_1h
        # self.seg_info = SegInfo(os.path.join(workspace, "seg_info.csv"))
        # self.regpair_info = np.load(os.path.join(workspace, "ATT_matrix_10.npy"))

    def load4sse(self, phase):
        data_path = self.workspace + "30/split/{}_set.json".format(phase)
        # combined info
        #     - o, d, t
        #     - coordinates of o and d
        #     - fastest path: node sequence
        #     - node coordinates sequence
        #     - region sequence
        #     - region spatial coordinates sequence
        #     - edge sequence with in_time and out_time
        with open(data_path, 'r') as f:
            data = json.load(f)
        return data

    def get_weight_info(self, phase):
        weight_path = self.workspace + "30/split/{}_weight.json".format(phase)
        with open(weight_path, 'r') as f:
            weight = json.load(f)
        return weight

    def get_regpair_info(self, workspace):
        return np.load(os.path.join(workspace, "ATT_matrix_10.npy"))

    # region info
    #   - coarse gird cords
    #   - fine grid cords
    #   - nodes in fine grids
    def get_region_info(self, workspace):
        # return np.load(os.path.join(workspace, "grids_info.json"), allow_pickle=True)
        with open(os.path.join(workspace, "grids_info.json"), "r") as f:
            data = json.load(f)
        return data


    # def get_weight_info(self, workspace):
    #     time_dense_file = workspace + "10/weight/grids_tdense_10.txt"
    #     pro_dis_file = workspace + "10/weight/pro_dis_10.txt"
    #
    #     time_dense = []
    #     pro_distances = []
    #
    #     with open(time_dense_file, 'r') as r_dense_in:
    #         for line in r_dense_in:
    #             index, value = map(float, line.strip().split())
    #             time_dense.append(value)
    #     print("Read grid time density file finished.")
    #
    #     with open(pro_dis_file, 'r') as p_dis_in:
    #         while True:
    #             size_line = p_dis_in.readline()
    #             if not size_line:
    #                 break
    #             size = int(size_line.strip())
    #             grids_vec = []
    #             for _ in range(size):
    #                 first, second = map(float, p_dis_in.readline().strip().split())
    #                 grids_vec.append((int(first), second))
    #             pro_distances.append(grids_vec)
    #     print("Read proximity distance file finished.")
    #
    #     return time_dense, pro_distances


class QryRegData(Dataset):
    def __init__(self, data, region_num, weights):
        self.data = data
        # self.region_data = region_data
        self.region_size = region_num
        self.weights = weights

    # def __init__(self, data, region_num):
    #     self.data = data
    #     # self.region_data = region_data
    #     self.region_size = region_num
    #     # self.weights = weights

    # def __getitem__(self, idx):
    #     odt, od_cords, node_seq, node_cords_seq, reg_seq, reg_cords_seq, edge_seq, uniq_reg, uniq_reg_cords = self.data[idx]
    #     label = self.get_onehot_target(uniq_reg, self.region_size)
    #     # o, d, t, o_reg, d_reg = self.get_padded_data(odt, reg_seq, self.region_size)
    #     o = odt[0]
    #     d = odt[1]
    #     t = odt[2]
    #     o_reg = reg_seq[0]
    #     d_reg = reg_seq[-1]
    #     regions = [i for i in range(self.region_size)]
    #     return o, d, t, o_reg, d_reg, label, np.array(regions)

    def __getitem__(self, idx):
        od, cost, node_seq, reg_seq = self.data[idx]
        label = self.get_onehot_target(reg_seq, self.region_size)
        # o, d, t, o_reg, d_reg = self.get_padded_data(odt, reg_seq, self.region_size)
        o = od[0]
        d = od[1]
        # t = 0
        o_reg = reg_seq[0]
        d_reg = reg_seq[-1]
        regions = [i for i in range(self.region_size)]
        weights = self.weights[idx]
        return o, d, o_reg, d_reg, label, np.array(regions), np.array(weights)


    def __len__(self):
        return len(self.data)

    def get_onehot_target(self, reg_seq, region_size):
        all_regions = [int(i) for i in range(region_size)]
        target = [1 if region in reg_seq else 0 for region in all_regions]
        return np.array(target)

    def get_padded_data(self, odt, reg_seq, region_size):
        pad_size = region_size
        padded_o = [odt[0]] + [0]*(pad_size - 1)
        padded_d = [odt[1]] + [0]*(pad_size - 1)
        padded_t = [odt[2]] + [0]*(pad_size - 1)
        padded_o_reg = [reg_seq[0]] + [0]*(pad_size - 1)
        padded_d_reg = [reg_seq[-1]] + [0]*(pad_size - 1)

        return padded_o, padded_d, padded_t, padded_o_reg, padded_d_reg


class MLP(nn.Module):
    def __init__(self, dim_input, dim_hidden, dim_output, dropout=0.1):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim_input, dim_hidden),
            nn.BatchNorm1d(dim_hidden, eps=1e-6),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_hidden, dim_output)
        )

    def forward(self, x):
        x = self.mlp(x)
        return x


class ODEncoder(nn.Module):
    def __init__(self, hparams):
        super(ODEncoder, self).__init__()
        self.hparams = hparams
        # Node2Vec feature: [node_num, 64]
        if self.hparams.use_node2vec:
            node2vec = torch.from_numpy(np.load(hparams.pretrained_node_emb))
            self.node2vec_feat = nn.Embedding.from_pretrained(node2vec, freeze=False)
        else:
            self.node2vec_feat = nn.Embedding(num_embeddings=hparams.node_num, embedding_dim=hparams.d_node)

    def forward(self, ori, dest):
        emb_o = self.node2vec_feat(ori)
        emb_d = self.node2vec_feat(dest)
        # code = torch.cat((emb_o, emb_d), dim=1)
        # return code
        return emb_o, emb_d


# Model Architecture
class SSRegionPred(nn.Module):
    def __init__(self, hparams):
        super(SSRegionPred, self).__init__()
        self.od_layer = ODEncoder(hparams)
        self.region_num = hparams.region_num
        # self.region_emb = nn.Embedding(hparams.region_num, hparams.d_r)
        # adopt initial region embedding and update it
        init_region_emb = torch.from_numpy(np.load(hparams.pretrained_region_emb))
        self.region_emb = nn.Embedding.from_pretrained(init_region_emb, freeze=False)

        # MLP2 with two layer
        self.mlp = MLP(hparams.d_m1, hparams.d_m2, hparams.d_m3, hparams.dropout)
        self.fc_layer1 = nn.Linear(1, hparams.batch_size, bias=False)
        self.fc_layer2 = nn.Linear(hparams.d_m4, hparams.d_node, bias=False)
        self.dense = nn.Linear(hparams.region_num, hparams.d_node, bias=False)
        # self.dense1 = nn.Linear(hparams.d_m3, hparams.region_num, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, o, d, o_reg, d_reg, regions, weights, train_phase=True):
        # o. d
        o_emb, d_emb = self.od_layer(o, d)

        # o_reg, d_reg
        # weight_dtype = self.fc_layer1.weight.dtype
        # o_reg1 = o_reg.view(-1, 1).to(weight_dtype)
        # d_reg1 = d_reg.view(-1, 1).to(weight_dtype)
        # o_reg_emb = self.fc_layer1(o_reg1)
        # d_reg_emb = self.fc_layer1(d_reg1)

        o_reg_emb = self.region_emb(o_reg)
        d_reg_emb = self.region_emb(d_reg)

        # combine o and o_reg, d and d_reg
        o_comb = torch.cat((o_emb, o_reg_emb), dim=1)
        d_comb = torch.cat((d_emb, d_reg_emb), dim=1)

        # o_comb = o_emb
        # d_comb = d_emb

        # combine o and d
        query_embs = torch.cat((o_comb, d_comb), dim=1)
        query_embs = self.mlp(query_embs)

        region_embs = self.region_emb(regions)

        # concatenate query embedding with each candidate region embedding
        query_embs = query_embs.unsqueeze(1)
        query_embs_exp = query_embs.expand(-1, self.region_num, -1)
        inputs = torch.cat((query_embs_exp, region_embs), dim=-1)

        output = self.fc_layer2(inputs)
        output = torch.mean(output, dim=-1, keepdim=True)
        output = self.sigmoid(output).squeeze(dim=-1)  # 1000, 100, 64

        if train_phase:
            return output
        else:
            g_prob, g_ssgrid = torch.max(output, dim=1)
            g_ssgrid = torch.sum(F.one_hot(g_ssgrid, output.shape[1]) * regions, dim=1)
            return g_ssgrid

        # sub_w_ = self.dense1.weight[regions]
        # sub_b_ = self.dense1.bias[regions]
        # output = torch.matmul(sub_w_, inputs.unsqueeze(-1)).squeeze(-1) + sub_b_
        #
        # if train_phase:
        #     return output
        # else:
        #     g_prob, g_kseg = torch.max(output, dim=1)
        #     g_kseg = torch.sum(F.one_hot(g_kseg, output.shape[1]) * regions, dim=1)
        #     return g_kseg