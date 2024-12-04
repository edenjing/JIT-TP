#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
Tasks:
1. read, select and convert trajectories into queries  ---- useless
2. store necessary information:
    - o, d, t (with coordinates)
    - fastest path from o to d: road sequence, region sequence, edge sequence
3. write the information into txt file
    - combined info w.r.t. queries
    - region info: region id, min(max) long(lat)
-----------------------------------------------------------------------------
Implement by C++:
1. read generated query_nodes at last step
2. perform A* algorithm to find the fastest path for each query
3. store the examined edges (regions) to construct ground truth search space
Generated files:
    - query_info/queries.txt/tTimes.txt/paths.txt/eGrids.txt/att_matrix.txt
-----------------------------------------------------------------------------
'''

import time
from constants import *
import json
import os


def read_map(new_path, raw_path):
    roads_path = new_path + "/new_beijingRoadNew.txt"
    coords_path = raw_path + "/beijingNodeNew"

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
    print("Read map finished.")
    return roads_dict, nodes_dict


def read_grids_info(grids_info_path):
    with open(grids_info_path, 'r') as f:
        grids_info = json.loads(f.read())
    print("Read grids_info_{}.json".format(coarse_gra))
    return grids_info


def read_query_info(query_info_path):
    queries = []
    tTimes = []
    paths = []
    eEdges = []

    with open(query_info_path, 'r') as f:
        for line in f:
            parts = line.strip().split('|')  # Separator between path and eEdges
            query_data = parts[0].strip().split()
            edge_data = parts[1].strip().split() if len(parts) > 1 else []

            # extract query, tTime, and path
            queries.append((int(query_data[0]), int(query_data[1])))
            tTimes.append(int(query_data[2]))
            path_end = query_data.index('|') if '|' in query_data else len(query_data)
            paths.append([int(x) for x in query_data[3:path_end]])

            # extract eEdges
            eEdges.append([int(x) for x in edge_data])

    print("Read query_info_{}.txt finished.".format(coarse_gra))

    return queries, tTimes, paths, eEdges


# convert paths in edge sequence into paths in node sequence
def node_path_convert(Paths, roads_dict):
    new_Paths = []
    for path in Paths:
        new_path = []
        for i in range(len(path)):
            road_info = path[i]
            road_id = road_info[0]
            if i == 0:
                new_path.append(roads_dict[road_id]["ID1"])
            new_path.append(roads_dict[road_id]["ID2"])

        new_Paths.append(new_path)
    return new_Paths


def grid_path_convert(Paths, grid_coords, nodes_dict):
    Grid_Paths = []

    for path in Paths:
        grid_seq = map_nodeseq_to_gridseq(path, grid_coords, nodes_dict)
        Grid_Paths.append(grid_seq)
    return Grid_Paths


def map_nodeseq_to_gridseq(path, grid_coords, nodes_dict):
    grid_sequence = []

    for node in path:
        node_long = nodes_dict[node]["long"]
        node_lat = nodes_dict[node]["lat"]

        # find the grid contains the node
        for grid_info in grid_coords:
            label_id, grid_min_long, grid_max_long, grid_min_lat, grid_max_lat = grid_info[0]
            if grid_min_long <= node_long <= grid_max_long and grid_min_lat <= node_lat <= grid_max_lat:
                grid_sequence.append(label_id)
                break

    return grid_sequence


# combine all necessary info together
#     - o, d, t (with coordinates)
#     - fastest path from o to d: node sequence (with coordinates)
#     - region sequence (with region spatial coordinates)
#     - edge sequence (with in_time and out_time)
def combine_info(Queries, Node_Paths, Grid_Paths, nodes_dict, grid_cords, Paths):
    Combined_Info_List = []

    for query in Queries:
        Info = []
        Info.append(query)  # o, d, t

        ID1 = query[0]
        ID2 = query[1]
        OD_cords = [[nodes_dict[ID1]["long"], nodes_dict[ID1]["lat"]], [nodes_dict[ID2]["long"], nodes_dict[ID2]["lat"]]]
        Info.append(OD_cords)  # coordinates of o and d

        path = Node_Paths[Queries.index(query)]
        path_cords = []
        for node in path:
            node_cords = [nodes_dict[node]["long"], nodes_dict[node]["lat"]]
            path_cords.append(node_cords)
        Info.append(path)  # node sequence
        Info.append(path_cords)   # coordinates of node sequence

        grid_seq = Grid_Paths[Queries.index(query)]
        grid_seq_cords = []
        for grid in grid_seq:
            grid_seq_cords.append(list(grid_cords[grid][0][-4:]))
        Info.append(grid_seq)  # grid sequence
        Info.append(grid_seq_cords)  # cords of grid sequence

        # edge sequence
        edge_seq = Paths[Queries.index(query)]
        Info.append(edge_seq)

        Combined_Info_List.append(Info)
    return Combined_Info_List


def write_files(Queries, combined_info, store_path):
    query_path = store_path + "queries_raw.txt"
    query_info_path = store_path + "query_info.json"

    with open(query_path, 'w') as f1:
        for query in Queries:
            line = ' '.join(map(str, query))
            f1.write(line + '\n')
    print("Saved queries_raw.txt.")

    # write departure times and paths
    with open(query_info_path, 'w') as f2:
        json_query_info = json.dumps(combined_info)
        f2.write(json_query_info)
    print("Saved query_info.json.")


if __name__ == '__main__':
    start0 = time.time()
    # roads_dict, nodes_dict = read_map(bj_new_generated_path, bj_traj_path)

    end0 = time.time()
    # grids_info = read_grids_info(beijing_path + "{}/grids_info_{}.json".format(coarse_gra, coarse_gra))

    queries, tTimes, paths, eEdges = read_query_info(beijing_path + "30/query_info_30.txt")
    print("Read data finished. Cost Time:", end0 - start0, 's')

    # Path Convert
    # # node sequence
    # start1 = time.time()
    # Node_Paths = node_path_convert(Q_Trajs, roads_dict)
    # # grid sequence
    # Grid_Paths = grid_path_convert(Node_Paths, grids_info, nodes_dict)
    # end1 = time.time()
    # print("Convert path finished. Cost Time:", end1 - start1, 's')
    #
    # # combine all info
    # Combined_Info = combine_info(Queries, Node_Paths, Grid_Paths, nodes_dict, grids_info, Q_Trajs)
    # write_files(Queries, Combined_Info, beijing_path)

    # Queries, Q_Trajs = read_json_trajs(bj_new_generated_path, traj_json_day, roads_dict, min_roadNum)

    print("All finished.")



# -------------------------------------------------------------------
# def read_json_trajs(new_path, day, roads_dict, min_roadNum):
#     Queries = []
#     Q_Trajs = []
#
#     json_path = new_path + "/Trajs_json/"
#     for i in range(1, day):
#         filename = "Day_" + str(day) + "_trajs.json"
#         read_path = os.path.join(json_path, filename)
#
#         if os.path.exists(read_path):
#             with open(read_path, 'r') as file:
#                 trajs = json.load(file)
#                 for traj in trajs:
#                     # traj satisfies the length and connectivity constraint
#                     if check_connectivity(traj, roads_dict, min_roadNum) == 0:
#                         ID1 = roads_dict[traj[0][0]]["ID1"]
#                         ID2 = roads_dict[traj[-1][0]]["ID2"]
#                         dep_time = traj[0][1]
#                         Queries.append([ID1, ID2, dep_time])
#                         Q_Trajs.append(traj)
#         else:
#             print("File {} does not exist.".format(filename))
#     # print("Read and select trajs finished.")
#     return Queries, Q_Trajs


# def check_connectivity(traj, roads_dict, min_roadNum):
#     # Single road or empty trajectory is considered connected
#     if len(traj) <= min_roadNum:
#         return 1
#
#     for i in range(1, len(traj)):
#         prev_id = traj[i-1][0]
#         curr_id = traj[i][0]
#         if roads_dict[prev_id]['ID2'] != roads_dict[curr_id]['ID1']:
#             return 2
#     return 0
