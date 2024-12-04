#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
Tasks:
1. region preprocess
    - split the entire road network by coarse granularity
    - store the spatial information of each region: coordinates
    - split each region by fine granularity 
    - store the index of each fine-region
2. generate random queries
    - select nodes in each fine region to generate queries and ensure the variety of sampling

Output Files:
1. grids_info_{}.json
2. query_nodes_{}.txt
3. ATT_matrix_{}.npy -- no need any more
'''

import random
from constants import *
import json
import time
from collections import defaultdict
import heapq
import numpy as np


def write_SPbyTime(store_path):
    # read SP
    SP_Data = {}
    with open(store_path + "new_BJSP.txt", 'r') as f1:
        for line in f1:
            edge_info = line.strip().split()
            edge_id = int(edge_info[0])

            for i in range(2, len(edge_info), 2):
                timestamp = int(edge_info[i])

                if timestamp not in SP_Data:
                    SP_Data[timestamp] = {}

                # travel time
                time = int(edge_info[i+1])
                SP_Data[timestamp][edge_id] = time
    print("Read new_BJSP.txt finished.")

    # write sp by time
    with open(store_path + "new_BJSPbyTime.txt", 'w') as f2:
        for timestamp, sp_info in SP_Data.items():
            line = f"{timestamp}"
            for edge_id, time in sp_info.items():
                line += f", {edge_id} {time}"
            line += "\n"
            f2.write(line)
    print("Write new_BJSPbyTime.txt finished.")


def read_data(new_path, raw_path):
    # with open(new_path + "/trajectories.json", "r") as f1:
    #     Trajs = json.load(f1)

    roads_path = raw_path + "new_beijingRoadNew.txt"
    coords_path = raw_path + "beijingNodeNew"
    sp_path = new_path + "new_BJSPbyTime.txt"

    roads_dict = {}  # road_id, ID1, ID2, length
    nodes_dict = {}  # node_id, longitude, latitude

    with open(roads_path, 'r') as f2:
        first_line = f2.readline().strip(" ")
        edge_num = first_line.split(" ")

        for line in f2:
            road_info = line.strip(" ")
            if road_info:
                road_id, _, length, ID1, ID2, speed_limit = road_info.split(' ')
                if road_id not in roads_dict:
                    roads_dict[int(road_id)] = {}
                roads_dict[int(road_id)]["ID1"] = int(ID1)
                roads_dict[int(road_id)]["ID2"] = int(ID2)
                roads_dict[int(road_id)]["len"] = int(length)
                roads_dict[int(road_id)]["sl"] = int(speed_limit)

    with open(coords_path, 'r') as f3:
        first_line = f3.readline().strip(" ")
        _, min_lat, max_lat, min_long, max_long = first_line.split("\t")

        for line in f3:
            node_info = line.strip("\t")
            if node_info:
                values = node_info.split()
                node_id = values[0]
                if node_id not in nodes_dict:
                    nodes_dict[int(node_id)] = {}
                nodes_dict[int(node_id)]["long"] = float(values[3])
                nodes_dict[int(node_id)]["lat"] = float(values[2])

    # read the whole file
    # SP_Data = {}
    # with open(sp_path, 'r') as f4:
    #     for line in f4:
    #         sp_info = line.strip().split(",")
    #         timestamp = int(sp_info[0])
    #         for item in sp_info[1:]:
    #             edge_id, time = item.strip().split()
    #             if timestamp not in SP_Data:
    #                 SP_Data[timestamp] = {}
    #             SP_Data[timestamp][edge_id] = time

    # return roads_dict, nodes_dict, SP_Data
    return roads_dict, nodes_dict


# raw BJSP
def get_sp_by_tsp(sp_path, edge_num, tsp):
    sp_fp = sp_path + "BJSP.txt"
    SP_Data = {}

    with open(sp_fp, 'r') as f:
        for line in f:
            sp_info = line.strip().split(" ")
            edge_id = int(sp_info[0])
            if edge_id not in SP_Data and edge_id < edge_num:
                SP_Data[edge_id] = {}
                for i in range(2, len(sp_info), 2):
                    timestamp = int(sp_info[i])
                    if timestamp == tsp:
                        time = int(sp_info[i+1])
                        SP_Data[edge_id] = time
                        break
    # print(SP_Data)
    return SP_Data


# split road network by coarse and fine granularity
# def split_network(min_long, max_long, min_lat, max_lat, coarse_gra, fine_gra):
#     grid_coordinates = []
#
#     # Calculate the number of grids in longitude and latitude directions for coarse granularity
#     num_grids_long_coarse = float((max_long - min_long) / coarse_gra)
#     num_grids_lat_coarse = float((max_lat - min_lat) / coarse_gra)
#
#     grid_id = 0
#     for i in range(coarse_gra):
#         for j in range(coarse_gra):
#             # Calculate the coordinates for each coarse grid
#             grid_min_long_coarse = min_long + j * num_grids_long_coarse
#             grid_max_long_coarse = min_long + (j + 1) * num_grids_long_coarse
#             grid_min_lat_coarse = min_lat + i * num_grids_lat_coarse
#             grid_max_lat_coarse = min_lat + (i + 1) * num_grids_lat_coarse
#
#             # Append the coarse grid coordinates
#             coarse_grid_coordinates = (grid_id, grid_min_long_coarse, grid_max_long_coarse, grid_min_lat_coarse, grid_max_lat_coarse)
#             grid_id += 1
#
#             fine_grid_coordinates = []
#             # Calculate the number of grids in longitude and latitude directions for fine granularity
#             num_grids_long_fine = float((grid_max_long_coarse - grid_min_long_coarse) / fine_gra)
#             num_grids_lat_fine = float((grid_max_lat_coarse - grid_min_lat_coarse) / fine_gra)
#             inner_grid_id = 0
#             for m in range(fine_gra):
#                 for n in range(fine_gra):
#                     # Calculate the coordinates for each fine grid within the current coarse grid
#                     grid_min_long_fine = grid_min_long_coarse + n * num_grids_long_fine
#                     grid_max_long_fine = grid_min_long_coarse + (n + 1) * num_grids_long_fine
#                     grid_min_lat_fine = grid_min_lat_coarse + m * num_grids_lat_fine
#                     grid_max_lat_fine = grid_min_lat_coarse + (m + 1) * num_grids_lat_fine
#
#                     # Append the fine grid coordinates
#                     fine_grid_coordinates.append((inner_grid_id, grid_min_long_fine, grid_max_long_fine, grid_min_lat_fine, grid_max_lat_fine))
#                     inner_grid_id += 1
#
#             combined_grid_coordinates = [coarse_grid_coordinates, fine_grid_coordinates]
#             grid_coordinates.append(combined_grid_coordinates)
#
#     return grid_coordinates


def split_network(min_long, max_long, min_lat, max_lat, coarse_gra, fine_gra):
    grid_coordinates = []
    adjacent_ids = {}

    # Calculate the number of grids in longitude and latitude directions for coarse granularity
    num_grids_long_coarse = float((max_long - min_long) / coarse_gra)
    num_grids_lat_coarse = float((max_lat - min_lat) / coarse_gra)

    grid_id = 0
    for i in range(coarse_gra):
        for j in range(coarse_gra):
            # Calculate the coordinates for each coarse grid
            grid_min_long_coarse = min_long + j * num_grids_long_coarse
            grid_max_long_coarse = min_long + (j + 1) * num_grids_long_coarse
            grid_min_lat_coarse = min_lat + i * num_grids_lat_coarse
            grid_max_lat_coarse = min_lat + (i + 1) * num_grids_lat_coarse

            # Append the coarse grid coordinates
            coarse_grid_coordinates = (
            grid_id, grid_min_long_coarse, grid_max_long_coarse, grid_min_lat_coarse, grid_max_lat_coarse)

            # Calculate adjacent grid IDs for coarse grid
            adj_ids = []
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    adj_i, adj_j = i + di, j + dj
                    if 0 <= adj_i < coarse_gra and 0 <= adj_j < coarse_gra:
                        adj_ids.append(adj_i * coarse_gra + adj_j)

            adjacent_ids[grid_id] = adj_ids

            fine_grid_coordinates = []
            # Calculate the number of grids in longitude and latitude directions for fine granularity
            num_grids_long_fine = float((grid_max_long_coarse - grid_min_long_coarse) / fine_gra)
            num_grids_lat_fine = float((grid_max_lat_coarse - grid_min_lat_coarse) / fine_gra)
            inner_grid_id = 0
            for m in range(fine_gra):
                for n in range(fine_gra):
                    # Calculate the coordinates for each fine grid within the current coarse grid
                    grid_min_long_fine = grid_min_long_coarse + n * num_grids_long_fine
                    grid_max_long_fine = grid_min_long_coarse + (n + 1) * num_grids_long_fine
                    grid_min_lat_fine = grid_min_lat_coarse + m * num_grids_lat_fine
                    grid_max_lat_fine = grid_min_lat_coarse + (m + 1) * num_grids_lat_fine

                    # Append the fine grid coordinates
                    fine_grid_coordinates.append(
                        (inner_grid_id, grid_min_long_fine, grid_max_long_fine, grid_min_lat_fine, grid_max_lat_fine))
                    inner_grid_id += 1

            combined_grid_coordinates = [coarse_grid_coordinates, fine_grid_coordinates]
            grid_coordinates.append(combined_grid_coordinates)

            grid_id += 1

    return grid_coordinates, adjacent_ids


# get all nodes in each finer region
def write_grids_info(grid_coordinates, adjacent_ids, nodes_dict, write_path):
    updated_grids_info = []
    for grid in grid_coordinates:
        node_info = []
        for fine_grid in grid[1]:
            nodes = [node for node in nodes_dict if fine_grid[1] <= nodes_dict[node]["long"] < fine_grid[2] and fine_grid[3] <= nodes_dict[node]["lat"] < fine_grid[4]]
            node_info.append(nodes)
        updated_grids_info.append(grid + [node_info])

    # write grids info: spatial coordinates and inner nodes
    grids_info_path = write_path + "30/grids_info_{}.json".format(coarse_gra)
    adj_grids_path = write_path +"30/adj_grids_{}.json".format(coarse_gra)

    with open(grids_info_path, 'w') as f1:
        json_grids_info = json.dumps(updated_grids_info)
        f1.write(json_grids_info)
    print("Saved grids_info_{}.json.".format(coarse_gra))

    with open(adj_grids_path, 'w') as f2:
        json_adj_grids = json.dumps(adjacent_ids)
        f2.write(json_adj_grids)
    print("Saved adj_grids_{}.json.".format(coarse_gra))

    return updated_grids_info


def read_grids_info(grids_info_path):
    with open(grids_info_path, 'r') as f:
        grids_info = json.loads(f.read())
    print("Read grids_info_{}.json".format(coarse_gra))
    return grids_info


def write_grids_cords(grids_info, grids_cords_path):
    with open(grids_cords_path + "30/grids_cords_30.txt", "w") as f:
        for grid_info in grids_info:
            tuple_item = grid_info[0]
            line = ' '.join(map(str, tuple_item))
            f.write(line + "\n")
    print("Saved grids_cords_{}.txt.".format(coarse_gra))


# select nodes for each fine grid
def get_nodes_for_query(grids_info, node_num_in_fine_grid, write_path):
    nodes_for_query = []
    for grid in grids_info:
        nodes_list = grid[2]
        selected_nodes_list = []
        for nodes in nodes_list:
            if len(nodes) <= node_num_in_fine_grid:
                selected_nodes = nodes
            else:
                selected_nodes = random.sample(nodes, node_num_in_fine_grid)
            selected_nodes_list.extend(selected_nodes)
        nodes_for_query.append(selected_nodes_list)

    # write queries in txt
    query_path = write_path + "{}/query_nodes_{}.txt".format(coarse_gra, coarse_gra)
    with open(query_path, 'w') as f:
        for nodes in nodes_for_query:
            line = ' '.join(map(str, nodes))
            f.write(line+'\n')
    print("Saved query_nodes_{}.txt.".format(coarse_gra))
    return nodes_for_query


# generate paths for queries -- Dijkstra Algorithm
def get_tt_for_queries(node_lists, SP_Data, roads_dict, nodenum, write_path):
    # get speed profile and construct computation graph
    # dep_time = random.randint(1, max_dep_time)
    # SP = SP_Data[dep_time]
    # SP = SP_Data[0]
    # print("Get speed profile correctly.")
    # print("SP:", SP)

    # node num: 296710; edge num: 651748
    graph = defaultdict(list)
    for edge_id, edge_data in roads_dict.items():
        node1 = edge_data['ID1']
        node2 = edge_data['ID2']
        time = SP_Data[edge_id]  # speed profile (dict) from another file
        graph[node1].append((node2, int(time)))

    # compute average travel time iteratively
    matrix_dim = pow(coarse_gra, 2)
    ATT_matrix = [[0.0] * matrix_dim for _ in range(matrix_dim)]

    for i in range(matrix_dim):
        for j in range(matrix_dim):
            travel_times = []

            if len(node_lists[i]) == 0 or len(node_lists[j]) == 0:
                ATT_matrix[i][j] = 0
                continue
            else:
                for node_a in node_lists[i]:
                    for node_b in node_lists[j]:
                        travel_time = Dijkstra_Dis(graph, node_a, node_b, nodenum)
                        if travel_time == INF:
                            travel_time = 0
                        if travel_time != 0:
                            print("node_a:", node_a, "node_b:", node_b, "travel_time:", travel_time)
                        travel_times.append(travel_time)

                if len(travel_times) == 0:
                    avg_travel_time = 0
                else:
                    avg_travel_time = sum(travel_times) / len(travel_times)
                ATT_matrix[i][j] = avg_travel_time
    print("Generate ATT_matrix finished.")

    # count non-zero numbers in the matrix
    ATT_matrix = np.array(ATT_matrix)
    non_zero_count = np.count_nonzero(ATT_matrix)
    print("Non-zero count in ATT_matrix:", non_zero_count)

    # write ATT_matrix
    # ATT_matrix_path = write_path + "ATT_matrix.txt"
    # with open(ATT_matrix_path, 'w') as f:
    #     json_ATT_matrix = json.dumps(ATT_matrix)
    #     f.write(json_ATT_matrix)
    # print("Saved ATT_matrix.txt.")

    ATT_matrix_path = write_path + "{}/ATT_matrix_{}.npy".format(coarse_gra, coarse_gra)
    np.save(ATT_matrix_path, ATT_matrix)
    print("Saved ATT_matrix_{}.npy.".format(coarse_gra))


# compute the fastest travel time for a given pair of nodes
def Dijkstra_Dis(graph, ID1, ID2, nodenum):
    if ID1 == ID2:
        return 0

    pqueue = []  # priority queue
    heapq.heappush(pqueue, (0, ID1))

    distance = [INF] * (nodenum + 1)
    closed = [False] * (nodenum + 1)

    distance[ID1] = 0
    d = INF

    while pqueue:
        topNodeDis, topNodeID = heapq.heappop(pqueue)

        if topNodeID == ID2:
            d = distance[ID2]
            break

        closed[topNodeID] = True

        for NNodeID, NWeight in graph[topNodeID]:
            if not closed[NNodeID]:
                if distance[NNodeID] > NWeight + topNodeDis:
                    distance[NNodeID] = NWeight + topNodeDis
                    heapq.heappush(pqueue, (distance[NNodeID], NNodeID))
    return d


if __name__ == '__main__':
    # write_SPbyTime(bj_new_generated_path)
    start0 = time.time()
    # raw: new generated sp
    # roads_dict, nodes_dict, SP_Data = read_data(bj_new_generated_path, beijing_raw_path)

    # new: raw BJ sp
    roads_dict, nodes_dict = read_data(bj_new_generated_path, beijing_raw_path)
    # SP_Data = get_sp_by_tsp(beijing_raw_path, 651748, sp_tsp)
    print("Get speed profile correctly.")

    # generate queries
    grid_coordinates, adjacent_ids = split_network(BJ_min_long, BJ_max_long, BJ_min_lat, BJ_max_lat, coarse_gra, fine_gra)
    write_grids_info(grid_coordinates, adjacent_ids, nodes_dict, beijing_path)

    # grids_info = read_grids_info(beijing_path + "30/grids_info_{}.json".format(coarse_gra))
    # write_grids_cords(grids_info, beijing_path)

    # end0 = time.time()
    # print("Generate Grids Info finished. Cost Time:", end0 - start0, 's')
    #
    # start1 = time.time()
    # nodes_for_query = get_nodes_for_query(grids_info, node_num_in_fine_grid, beijing_path)
    # # get_tt_for_queries(nodes_for_query, SP_Data, roads_dict, bj_nodeNum, beijing_path)  # Dijkstra
    #
    # end1 = time.time()
    # print("Generate nodes_for_query finished. Cost Time:", end1 - start1, 's')


#------------------------------------------------------#
# def get_nodes(grid_coordinates, nodes_dict):
#     updated_grids_info = grid_coordinates
#     for i in range(len(grid_coordinates)):
#         grid = grid_coordinates[i]
#         node_info = []
#         for fine_grid in grid[1]:
#             nodes = []
#             for node in nodes_dict:
#                 if nodes_dict[node]["long"] >= fine_grid[1] and \
#                         nodes_dict[node]["long"] < fine_grid[2] and \
#                         nodes_dict[node]["lat"] >= fine_grid[3] and \
#                         nodes_dict[node]["lat"] < fine_grid[4]:
#                     nodes.append(node)
#             node_info.append(nodes)  # node info contains all nodes
#         updated_grids_info[i].append(node_info)
#     return updated_grids_info
