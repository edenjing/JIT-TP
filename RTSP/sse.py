#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

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