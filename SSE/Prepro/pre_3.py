#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
Tasks:
1. read Files:
    a. map files to get roads_dict and nodes_dict
    b. grids_info_{}.json
    c. c++ generated files:
        - query_info/queries.txt/tTimes.txt/paths.txt/eGrids.txt/att_matrix.txt

2. label search space grids for train, valid and test queries
   - find the grids in the search space, label as 1
   - other grids in the road network, label as 0
'''

from constants import *
import json
import time
import random
import math


def read_query_info(store_path):
    queries_path = store_path + "/queries.txt"
    tTimes_path = store_path + "/tTimes.txt"
    paths_path = store_path + "/paths.txt"
    eGrids_path = store_path + "/eGrids.txt"

    queries = []
    tTimes = []
    paths = []
    eGrids = []

    # read queries
    with open(queries_path, 'r') as f:
        for line in f:
            ori, dest = map(int, line.strip().split())
            queries.append([ori, dest])

    # read tTimes
    with open(tTimes_path, 'r') as f:
        tTimes = [float(line.strip()) for line in f]

    # read paths
    with open(paths_path, 'r') as f:
        for line in f:
            path = list(map(int, line.strip().split()))
            paths.append(path)

    # read eGrids
    with open(eGrids_path, 'r') as f:
        for line in f:
            egrid = list(map(int, line.strip().split()))
            eGrids.append(egrid)

    # verify that all lists have the same length
    assert len(queries) == len(tTimes) == len(paths) == len(eGrids), "Mismatch in data lengths"

    # combine information for each query
    comb_info = [
        [queries[i], tTimes[i], paths[i], eGrids[i]]
        for i in range(len(queries))
    ]

    return comb_info


def split_train_test(comb_info, weight_info):
    # shuffle the data randomly
    random.shuffle(comb_info)

    # delete useless data - no path result
    new_comb_info = []
    for query_info in comb_info:
        if len(query_info[2]) != 0 and len(query_info[3]) != 0:
            new_comb_info.append(query_info)

    train_size = int(len(new_comb_info) * train_ratio)
    valid_size = int(len(new_comb_info) * valid_ratio)

    train_set = new_comb_info[:train_size]
    valid_set = new_comb_info[train_size:train_size+valid_size]
    test_set = new_comb_info[train_size+valid_size:]
    print("len of train/valid/test set:", len(train_set), len(valid_set), len(test_set))

    train_weight = weight_info[:train_size]
    valid_weight = weight_info[train_size: train_size+valid_size]
    test_weight = weight_info[train_size+valid_size:]

    return train_set, valid_set, test_set, train_weight, valid_weight, test_weight


def write_labeled_data(train_set, valid_set, test_set, store_path):
    train_set_path = store_path + "split/train_set.json"
    val_set_path = store_path + "split/val_set.json"
    test_set_path = store_path + "split/test_set.json"

    with open(train_set_path, 'w') as f1, open(val_set_path, 'w') as f2, open(test_set_path, 'w') as f3:
        json_train_set = json.dumps(train_set)
        f1.write(json_train_set)
        json_val_set = json.dumps(valid_set)
        f2.write(json_val_set)
        json_test_set = json.dumps(test_set)
        f3.write(json_test_set)
    print("Saved train/valid/test_set.json")


def get_weight_info(weight_path):
    time_dense_file = weight_path + "grids_tdense_30.txt"
    pro_dis_file = weight_path + "pro_dis_30.txt"

    time_dense = []
    pro_distances = []

    with open(time_dense_file, 'r') as r_dense_in:
        for line in r_dense_in:
            index, value = map(float, line.strip().split())
            time_dense.append(value)
    # print("Read grid time density file finished.")

    count = 0
    with open(pro_dis_file, 'r') as p_dis_in:
        inner_pro_dis = []
        for line in p_dis_in:
            # first, second = map(float, line.strip().split())
            dis = float(line.strip())
            inner_pro_dis.append(dis)
            count += 1

            if count == coarse_gra * coarse_gra:
                pro_distances.append(inner_pro_dis.copy())
                inner_pro_dis = []
                count = 0

    if inner_pro_dis:
        pro_distances.append(inner_pro_dis)

    print("Read grid time density and proximity distance file finished.")

    # process time density
    min_time_dense = min(time_dense)
    # new_time_dense = [min_time_dense / dis for dis in time_dense]
    # new_time_dense = [math.exp(-dis / region_pro_para) for dis in time_dense]

    # process proximity
    # new_pro_distances = []
    combined_result = []
    for query_weights in pro_distances:
        # min_dis = min(query_weights)
        # new_weights = [dis / min_dis for dis in query_weights]
        new_weights = [math.exp(-dis / region_pro_para) for dis in query_weights]
        # new_pro_distances.append(new_weights)

        # compute proximity distance with time density grids_tdense_10.txt
        results = [x * y for x, y in zip(new_weights, time_dense)]
        # new_results = []ß
        # for value in results:
        #     if(math.isnan(value)):
        #         value = 0.0
        #     new_results.append(value)
        combined_result.append(results)
    return combined_result


def write_weight_data(train_weight, valid_weight, test_weight, store_path):
    train_weight_path = store_path + "split/train_weight.json"
    val_weight_path = store_path + "split/val_weight.json"
    test_weight_path = store_path + "split/test_weight.json"
    with open(train_weight_path, 'w') as f1, open(val_weight_path, 'w') as f2, open(test_weight_path, 'w') as f3:
        json_train_set = json.dumps(train_weight)
        f1.write(json_train_set)

        json_val_set = json.dumps(valid_weight)
        f2.write(json_val_set)

        json_test_set = json.dumps(test_weight)
        f3.write(json_test_set)
    print("Saved train/valid/test_weight.json")

# def read_query_info(store_path):
#     grid_path = store_path + "grids_info.json"
#     query_path = store_path + "query_info.json"
#
#     with open(grid_path, 'r') as f1, open(query_path, 'r') as f2:
#         grids_info = json.loads(f1.read())
#         query_info = json.loads(f2.read())
#
#     return grids_info, query_info


# def split_train_test(train_ratio, valid_ratio, combined_info):
#     random.shuffle(combined_info)  # shuffle the data randomly
#
#     train_size = int(len(combined_info) * train_ratio)
#     valid_size = int(len(combined_info) * valid_ratio)
#
#     train_set = combined_info[:train_size]
#     valid_set = combined_info[train_size:train_size+valid_size]
#     test_set = combined_info[train_size+valid_size:]
#
#     train_set = get_gt_grids(train_set)
#     valid_set = get_gt_grids(valid_set)
#     test_set = get_gt_grids(test_set)
#
#     return train_set, valid_set, test_set


# get ground truth search space grids for each query (get unique grid ids)
# def get_gt_grids(query_set):
#     new_query_set = []
#
#     for query_info in query_set:
#         grid_seq = query_info[4]
#         grid_cords_seq = query_info[5]
#
#         unique_grids = list(set(grid_seq))
#         unique_cords_seq = [grid_cords_seq[grid_seq.index(grid)] for grid in unique_grids]
#
#         query_info.append(unique_grids)
#         query_info.append(unique_cords_seq)
#
#         new_query_set.append(query_info)
#
#     return new_query_set


# def store_labeled_train_test(train_set, valid_set, test_set):
#     train_set_path = beijing_path + "split/train_set.json"
#     val_set_path = beijing_path + "split/val_set.json"
#     test_set_path = beijing_path + "split/test_set.json"
#
#     with open(train_set_path, 'w') as f1, open(val_set_path, 'w') as f2, open(test_set_path, 'w') as f3:
#         json_train_set = json.dumps(train_set)
#         f1.write(json_train_set)
#
#         json_val_set = json.dumps(valid_set)
#         f2.write(json_val_set)
#
#         json_test_set = json.dumps(test_set)
#         f3.write(json_test_set)
#     print("Saved train/valid/test_set.json")


if __name__ == '__main__':
    start = time.time()
    # grid_cords, query_info = read_query_info(beijing_path)
    # train_set, val_set, test_set = split_train_test(train_ratio, valid_ratio, query_info)
    # store_labeled_train_test(train_set, val_set, test_set)

    comb_info = read_query_info(beijing_path + "30/query_info")
    weight_info = get_weight_info(beijing_path + "30/weight/")
    train_set, valid_set, test_set, train_weight, valid_weight, test_weight = split_train_test(comb_info, weight_info)
    write_labeled_data(train_set, valid_set, test_set, beijing_path + "30/")
    write_weight_data(train_weight, valid_weight, test_weight, beijing_path + "30/")

    end = time.time()

    print("Generating queries finished. Cost time:", (end - start) / 60, "min")
