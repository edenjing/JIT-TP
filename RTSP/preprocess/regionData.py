#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
Task:
1. prepare the global features: region-connection graph (adjacent regions) and time domain
2. prepare the local features: nodes, edges and speed profile data of each region
3. for each region, get the mapping of new2old_edge_id, old2new_edge_id and the adjacent edge ids
'''

import pickle
import os
import numpy as np

# read all nodes and edges under a grid granularity
def read_grids_info(fp, gridGra):
    fp_node = fp + str(gridGra) + '/grids_nodes_{}.txt'.format(gridGra)
    fp_edge = fp + str(gridGra) + '/grids_edges_{}.txt'.format(gridGra)

    grids_nodes = {}
    grids_edges = {}

    with open(fp_node, 'r') as fn:
        for line in fn:
            # Split line into values and convert to integers
            values = [int(x) for x in line.strip().split()]
            grid_id = values[0]
            nodes = values[1:]
            grids_nodes[grid_id] = nodes

    with open(fp_edge, 'r') as fe:
        for line in fe:
            values = [int(x) for x in line.strip().split()]
            # if len(values) < 2:
            #     continue
            grid_id = values[0]
            edges = values[1:]
            grids_edges[grid_id] = edges
    return grids_nodes, grids_edges


# extract grid info for specified grid list
# def get_info_for_regions(grids_nodes, grids_edges, grids):
#     grid_nodes = {}
#     grid_edges = {}
#     for grid_id in grids:
#         grid_nodes[grid_id] = grids_nodes[grid_id]
#         grid_edges[grid_id] = grids_edges[grid_id]
#     return grid_nodes, grid_edges


# get speed profile of given grids
# def get_speed_profile(fp, grids, gridGra):
#     gridsSP = {}  # grid_id: speed data [edge_id: time, edge_id: time,...]
#     for grid_id in grids:
#         gridsp = {}
#         fp_grid = fp + str(gridGra) + '/SP/{}.txt'.format(grid_id)
#
#         with open(fp_grid, 'r') as fg:
#             for line in fg:
#                 sp_info = line.strip().split(" ")
#                 edge_id = int(sp_info[0])
#                 if edge_id not in gridsp:
#                     gridsp[edge_id] = {}
#                     for i in range(1, len(sp_info)-1, 2):
#                         tsp = int(sp_info[i])
#                         time = int(sp_info[i + 1])
#                         # if tsp not in gridsp[edge_id]:
#                         #     gridsp[edge_id][tsp] = {}
#                         gridsp[edge_id][tsp] = time
#         gridsSP[grid_id] = gridsp
#     return gridsSP


# def get_speed_profile(fp, grids, gridGra, time_slot_size):
#     gridsSP = {}  # grid_id: numpy array [num_edges, num_timestamps]
#
#     for grid_id in grids:
#         fp_grid = fp + str(gridGra) + '/SP/{}.txt'.format(grid_id)
#
#         # First pass: get all edge IDs and find max edge ID
#         edge_ids = set()
#         with open(fp_grid, 'r') as fg:
#             for line in fg:
#                 sp_info = line.strip().split(" ")
#                 edge_id = int(sp_info[0])
#                 edge_ids.add(edge_id)
#
#         # Create a numpy array for this grid
#         # Shape: [num_edges, num_timestamps]
#         edge_list = sorted(list(edge_ids))
#         edge_to_idx = {edge_id: idx for idx, edge_id in enumerate(edge_list)}  # get new edge id
#         grid_data = np.zeros((len(edge_ids), time_slot_size))
#
#         # Second pass: fill in the speed values
#         with open(fp_grid, 'r') as fg:
#             for line in fg:
#                 sp_info = line.strip().split(" ")
#                 edge_id = int(sp_info[0])
#                 edge_idx = edge_to_idx[edge_id]
#
#                 # Fill in speed values for this edge
#                 for i in range(1, len(sp_info) - 1, 2):
#                     timestamp = int(sp_info[i])
#                     speed = int(sp_info[i + 1])
#                     if timestamp < time_slot_size:
#                         grid_data[edge_idx, timestamp] = speed
#
#         # Forward fill missing values (fill with previous valid value)
#         for edge_idx in range(grid_data.shape[0]):
#             last_valid = 0
#             for t in range(grid_data.shape[1]):
#                 if grid_data[edge_idx, t] == 0:
#                     grid_data[edge_idx, t] = last_valid
#                 else:
#                     last_valid = grid_data[edge_idx, t]
#
#         gridsSP[grid_id] = grid_data
#
#     return gridsSP


def get_speed_profile(fp, grids, gridGra, time_slot_size):
    gridsSP = {}

    for grid_id in grids:
        fp_grid = fp + str(gridGra) + '/SP/{}.txt'.format(grid_id)

        # Read all lines at once
        with open(fp_grid, 'r') as fg:
            lines = fg.readlines()

        # Process edge IDs and data in a single pass
        edge_data = {}
        for line in lines:
            sp_info = line.strip().split()
            edge_id = int(sp_info[0])

            # Initialize array for this edge if not exists
            if edge_id not in edge_data:
                edge_data[edge_id] = np.zeros(time_slot_size)

            # Process timestamps and speeds in chunks
            timestamps = np.array([int(sp_info[i]) for i in range(1, len(sp_info) - 1, 2)])
            speeds = np.array([int(sp_info[i]) for i in range(2, len(sp_info), 2)])

            # Filter valid timestamps
            valid_mask = timestamps < time_slot_size
            valid_timestamps = timestamps[valid_mask]
            valid_speeds = speeds[valid_mask]

            # Assign speeds to timestamps
            edge_data[edge_id][valid_timestamps] = valid_speeds

        # Convert dictionary to numpy array
        edge_ids = sorted(edge_data.keys())
        grid_data = np.zeros((len(edge_ids), time_slot_size))

        for idx, edge_id in enumerate(edge_ids):
            grid_data[idx] = edge_data[edge_id]

        # Forward fill missing values using vectorized operations
        mask = grid_data == 0
        idx = np.where(~mask, np.arange(mask.shape[1]), 0)
        np.maximum.accumulate(idx, axis=1, out=idx)
        grid_data = grid_data[np.arange(grid_data.shape[0])[:, None], idx]

        gridsSP[grid_id] = grid_data

    return gridsSP


def save_speed_profile(gridsSP, save_path):
    """Save the processed speed profile to a pickle file"""
    gsp_path = save_path + "gridsSP.pkl"
    with open(gsp_path, 'wb') as f:
        pickle.dump(gridsSP, f)



# get adjacent regions
def get_adjacent_regions(fp, gridGra):
    adj_grids = {}
    fp_ag = fp + str(gridGra) + '/grids_adj_grids_{}.txt'.format(gridGra)
    with open(fp_ag, 'r') as fa:
        for line in fa:
            values = [int(x) for x in line.strip().split()]
            # if len(values) < 2:  # Skip empty lines or invalid format
            #     continue
            gid = values[0]
            agids = values[1:]
            adj_grids[gid] = agids
    return adj_grids


def reigon_matrix(fp, adj_grids):
    num_regions = len(adj_grids)
    adj_matrix = np.zeros((num_regions, num_regions), dtype=int)

    # fill the matrix
    for i in range(num_regions):
        neighbors = adj_grids[i]
        for neighbor in neighbors:
            adj_matrix[i][neighbor] = 1
            adj_matrix[neighbor][i] = 1  # undirected graph

    np.savetxt(fp + 'region_matrix.txt', adj_matrix, fmt='%d')
    print('Region matrix saved to {}'.format(fp + 'region_matrix.txt'))


def get_adjacent_edges(fp):
    adj_edges = {}
    adj_edge_fp = fp + 'edge_adj_edges.txt'
    with open(adj_edge_fp, 'r') as fe:
        for line in fe:
            values = [int(x) for x in line.strip().split()]
            # if len(values) < 2:  # Skip empty lines or invalid format
            #     continue
            eid = values[0]
            aeids = values[1:]
            adj_edges[eid] = aeids

    return adj_edges


# for each region, get the mapping of new2old_edge_id, old2new_edge_id and the adjacent edge ids
# return: old2new_edge_dict, new2old_edge_dict, new_edge2edge_neighbor_dict
def region_edge_mapping(adj_edges, grids_edges, gridGra):
    old2new_edge_dicts = []
    new2old_edge_dicts = []
    new_edge2edge_neighbor_dicts = []

    # for each grid
    for i in range(gridGra*gridGra):
        old2new = {}
        new2old = {}
        new2adj_edge = {}

        # get grid edges
        grid_edges = grids_edges[i]
        # get adjacent edges for each edge in grids
        old_adj_edges = find_adj_edges_in_grid(grid_edges, adj_edges)

        # get new edge id
        for i in range(len(grid_edges)):
            edge_id = grid_edges[i]
            old2new[edge_id] = i
            new2old[i] = edge_id

        # convert old ids of adjacent edges to new ids
        for old_edge in grid_edges:
            a_edges = old_adj_edges[old_edge]  # old ids of adjacent edges
            new2adj_edge[old2new[old_edge]] = [old2new[o_eid] for o_eid in a_edges]

        old2new_edge_dicts.append(old2new)
        new2old_edge_dicts.append(new2old)
        new_edge2edge_neighbor_dicts.append(new2adj_edge)

    return old2new_edge_dicts, new2old_edge_dicts, new_edge2edge_neighbor_dicts


# For all old edges in the network
def find_adj_edges_in_grid(grid_edges, adj_edges):
    re_adj_edges = {}
    for edge_id in grid_edges:
        re_adj_edges[edge_id] = []
        adj_eids = adj_edges[edge_id]
        for agid in adj_eids:
            if agid in grid_edges:
                re_adj_edges[edge_id].append(agid)
            else:
                continue
    return re_adj_edges


def write_file(fp, old2new_edge_dicts, new2old_edge_dicts, new_edge2edge_neighbor_dicts):
    fp_pkl = fp + 'new_edges_dict.pk'
    with open(fp_pkl, "wb") as f:
        pickle.dump((old2new_edge_dicts, new2old_edge_dicts, new_edge2edge_neighbor_dicts), f)
    print("Write files done.")


if __name__ == '__main__':
    gridGra = 10
    root_path = '/home/edenjingzhao/RegionData/'
    # grids_nodes, grids_edges = read_grids_info(root_path, gridGra)
    # adj_grids = get_adjacent_regions(root_path, gridGra)
    # adj_edges = get_adjacent_edges(root_path)
    # old2new_edge_dicts, new2old_edge_dicts, new_edge2edge_neighbor_dicts = region_edge_mapping(adj_edges, grids_edges, gridGra)

    project_path = '/data5/edenjingzhao/RTSP/data/Beijing/{}/'.format(gridGra)
    # convert and write region graph
    # reigon_matrix(project_path, adj_grids)
    Grids = [i for i in range(gridGra*gridGra)]
    gridsSP = get_speed_profile(root_path, Grids, gridGra, 288)  # dict
    save_speed_profile(gridsSP, project_path)
    # write_file(project_path, old2new_edge_dicts, new2old_edge_dicts, new_edge2edge_neighbor_dicts)
    print("End.")

