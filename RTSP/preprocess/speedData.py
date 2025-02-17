#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
Task:
1. prepare the speed data by hyperparameters: prediction_horizon
2. split the data into train, validation and test sets
'''

import torch
import numpy as np
import pickle
from torch import nn
from torch.utils.data import Dataset, DataLoader
from preprocess.regionData import get_speed_profile, read_grids_info, get_adjacent_regions

def load_data(speed_path, region_matrix_path, region_edges_path, gridGra, time_slot_size, edge_mapping_path):
    """
    Load data from txt files

    Expected file formats:
    - speed_data.txt: each row represents one edge, each column is a timestamp
    - region_edges.txt: each row format: "region_id edge1_id edge2_id edge3_id ..."
    - region_matrix.txt: adjacency matrix, each row represents connections of one region
    """
    # load adjacent region list
    # adjacent_regions = get_adjacent_regions(adj_regions_path, gridGra)

    # load region matrix (adjacent matrix)
    matrix_path = region_matrix_path + '{}/region_matrix.txt'.format(gridGra)
    region_matrix = np.loadtxt(matrix_path, dtype=int)  # ndarray
    
    # load region mapping
    _, regions_edges = read_grids_info(region_edges_path, gridGra)  # dict

    # load speed data for each region
    # Note: speed data for all grids here
    # Grids = [i for i in range(gridGra*gridGra)]
    # speed_data = get_speed_profile(speed_path, Grids, gridGra, time_slot_size)  # dict
    gridsSP_path = speed_path + '{}/gridsSP.pkl'.format(gridGra)
    with open(gridsSP_path, 'rb') as f:
        speed_data = pickle.load(f)

    # load: old2new_edge_dicts, new2old_edge_dicts, new_edge2edge_neighbor_dicts
    edge_map_path = edge_mapping_path + '{}/new_edges_dict.pk'.format(gridGra)
    old2new_edge_dicts, new2old_edge_dicts, new_edge2edge_neighbor_dicts = pickle.load(open(edge_map_path, "rb"))

    return speed_data, regions_edges, region_matrix, old2new_edge_dicts, new2old_edge_dicts, new_edge2edge_neighbor_dicts


class TrafficDataset(Dataset):
    def __init__(self,
                 speed_data: dict,  # Dictionary of region_id -> speed data array
                 region_mapping: dict,  # region_id -> list of edge indices
                 # region_matrix: np.ndarray,  # [num_regions, num_regions]
                 hist_len: int,
                 pred_len: int,
                 split_start: int,  # start index of this split
                 split_end: int):  # end index of this split

        self.speed_data = {region_id: torch.FloatTensor(data)
                          for region_id, data in speed_data.items()}
        self.region_mapping = region_mapping
        # self.region_matrix = torch.FloatTensor(region_matrix)
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.split_start = split_start
        self.split_end = split_end

    def __len__(self):
        # use first region's data to determine length
        # first_region_data = next(iter(self.speed_data.values()))
        return self.split_end - self.split_start - self.hist_len - self.pred_len + 1

    def __getitem__(self, idx):
        # actual starting index in the time dimension
        t = idx + self.split_start

        # historical data and ground truth for each region
        region_hist_data = {}
        region_future_data = {}

        for region_id in self.speed_data.keys():
            # get speed data for edges in this region
            region_edges_hist = self.speed_data[region_id][:,
                                t:t + self.hist_len]  # [num_edges_in_region, hist_len]
            region_edges_future = self.speed_data[region_id][:,
                                  t + self.hist_len:t + self.hist_len + self.pred_len]  # [num_edges_in_region, pred_len]

            region_hist_data[region_id] = region_edges_hist
            region_future_data[region_id] = region_edges_future

        return {
            'region_hist': region_hist_data,  # Dict[region_id -> Tensor[num_edges_in_region, hist_len]]
            'region_future': region_future_data  # Dict[region_id -> Tensor[num_edges_in_region, pred_len]]
            # 'region_matrix': self.region_matrix  # [num_regions, num_regions]
        }
