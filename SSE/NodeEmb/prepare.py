#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from args import make_args
from graph_embedding import *
import geopandas as gpd

if __name__ == '__main__':
    opt = make_args()
    print(opt)

    # 1 prepare data
    print("==========1 Prepare data from train.==========")
    # get_road_graph(opt)
    # print_graph(opt)

    # 2 label key search space for training
    print("==========2 Data are already labeled.==========")  # find the ground truth grids for each query
    # Note: data are already labeled

    # 3 node2vec embedding
    print("==========3 Node2Vec Embedding for road graph.==========")
    gen_node2vec_emb(opt)

    print("Prepare Workspace Finished!")
