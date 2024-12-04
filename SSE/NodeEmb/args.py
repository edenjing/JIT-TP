#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import math
import multiprocessing
from argparse import ArgumentParser


def make_args():
    parser = ArgumentParser()

    # Beijing
    parser.add_argument('--workspace', type=str, default="/data5/edenjingzhao/QDTP/data/BJData/bj_data/")
    parser.add_argument('--road_file', type=str, default="/data5/TrajectoryData/BeijingTrajectoryMatched/NewGeneratedFiles/new_beijingRoadNew.txt")
    parser.add_argument('--node_file', type=str, default="/data5/TrajectoryData/BeijingTrajectoryMatched/beijingNodeNew")
    parser.add_argument('-node_num', type=int, default=296710)
    parser.add_argument("-threadnum", type=int, default=int(math.ceil(multiprocessing.cpu_count() * 0.6)))

    # parser.add_argument('--train_file', type=str, default="data/BJData/bj_data/traj_train_20.csv")
    # parser.add_argument('--valid_file', type=str, default="data/BJData/bj_data/traj_valid.csv")
    # parser.add_argument('--test_file', type=str, default="data/BJData/bj_data/traj_test.csv")

    # for node2vec
    parser.add_argument('-walk_len', type=int, default=30)
    parser.add_argument('-num_walks', type=int, default=25)
    parser.add_argument('-p', type=float, default=1.0)
    parser.add_argument('-q', type=float, default=1.0)
    parser.add_argument('-emb_dim', type=int, default=64)
    parser.add_argument('-window', type=int, default=5)
    parser.add_argument('-epoch', type=int, default=10)

    # parser.add_argument('-label_size', type=int, default=30)
    # parser.add_argument('-mask_size', type=int, default=100)
    # parser.add_argument('-chunk_size', type=int, default=10000)

    args, unknown = parser.parse_known_args()

    return args

