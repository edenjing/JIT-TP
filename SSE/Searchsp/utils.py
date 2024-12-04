#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import time
import numpy as np
from torch.utils.data import Dataset
import torch

class Dict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__



def dict_to_object(dictObj):
    if not isinstance(dictObj, dict):
        return dictObj
    inst = Dict()
    for k, v in dictObj.items():
        inst[k] = dict_to_object(v)
    return inst


def to_torch_long(in_list, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    return torch.LongTensor(in_list).to(device)  # 64-bit integer


def list_to_tensor(data):
    # o, d, t, o_reg, d_reg, label, regions = data
    # o = to_torch_long(o)
    # d = to_torch_long(d)
    # t = to_torch_long(t)
    # o_reg = to_torch_long(o_reg)
    # d_reg = to_torch_long(d_reg)
    # label = to_torch_long(label)
    # regions = to_torch_long(regions)

    o, d, t, o_reg, d_reg, label, regions = data
    o = torch.tensor(o)
    d = torch.tensor(d)
    t = torch.tensor(t)
    o_reg = torch.tensor(o_reg)
    d_reg = torch.tensor(d_reg)
    label = torch.tensor(label)
    regions = torch.tensor(regions)

    return o, d, t, o_reg, d_reg, label, regions

        # dataloaders = []
        #     for traj in type_item:
        #         traj_input = data_to_device(traj)
        #
        #         traj, next_truth, des_truth, len_pad, length, \
        #         time_up, time_low, time_up_diff, time_low_diff, \
        #         dis_up, dis_low, dis_up_diff, dis_low_diff = traj_input
        #
        #         # Perform TrajDataLoader
        #         DATA = TrajDataLoader(traj, len_pad, length, time_up, time_low, time_up_diff, time_low_diff,
        #                                   dis_up, dis_low, dis_up_diff, dis_low_diff, next_truth, des_truth)
        #
        #         DATA_LOADER = DataLoader(DATA, batch_size=1)
        #
        #         dataloaders.append(DATA_LOADER)

# def DataOperation(data):
#     Dataloaders = []
#
#     for data_item in data:

