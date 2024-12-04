#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
Generate node2vec embeddings.
'''

import argparse
import networkx as nx
from tqdm import tqdm
import pickle
import os
import numpy as np
from ge.models.node2vec import Node2Vec
import time


def read_roads_and_nodes(args):
    roads_dict = {}  # road id, direction, length, ID1, ID2, speed limit
    nodes_dict = {}  # node id, longitude, latitude

    with open(args.road_file, 'r') as f1:
        first_line = f1.readline().strip(" ")
        edge_num = first_line.split(" ")

        for line in f1:
            road_info = line.strip(" ")
            if road_info:
                road_id, _, length, ID1, ID2, speed_limit = road_info.split(' ')
                if road_id not in roads_dict:
                    roads_dict[int(road_id)] = {}
                roads_dict[int(road_id)]["ID1"] = int(ID1)
                roads_dict[int(road_id)]["ID2"] = int(ID2)
                roads_dict[int(road_id)]["len"] = int(length)
                roads_dict[int(road_id)]["sl"] = int(speed_limit)

    with open(args.node_file, 'r') as f2:
        first_line = f2.readline().strip(" ")
        _, min_lat, max_lat, min_long, max_long = first_line.split("\t")

        for line in f2:
            node_info = line.strip("\t")
            if node_info:
                values = node_info.split()
                node_id = values[0]
                if node_id not in nodes_dict:
                    nodes_dict[int(node_id)] = {}
                nodes_dict[int(node_id)]["long"] = float(values[3])
                nodes_dict[int(node_id)]["lat"] = float(values[2])

    adj_nodes = find_adj_nodes(roads_dict, nodes_dict)
    return roads_dict, nodes_dict, adj_nodes


def find_adj_nodes(roads_dict, nodes_dict):
    node_ids = list(nodes_dict.keys())

    adj_nodes = {}
    for node_id in node_ids:
        adj_nodes[node_id] = []

    for road_id in roads_dict.keys():
        ID1 = roads_dict[road_id]["ID1"]
        ID2 = roads_dict[road_id]["ID2"]
        len = roads_dict[road_id]["len"]

        if ID2 not in adj_nodes[ID1]:
            adj_nodes[ID1].append(ID2)

    return adj_nodes


def find_length_by_nodes(roads_dict, nodeID1, nodeID2):
    for road_data in roads_dict.values():
        ID1 = road_data["ID1"]
        ID2 = road_data["ID2"]
        length = road_data["len"]

        if ID1 == nodeID1 and ID2 == nodeID2:
            return length

    print("{}-{} road length not found.".format(nodeID1, nodeID2))
    return 0


def get_road_graph(args):
    roads_dict, nodes_dict, adj_nodes = read_roads_and_nodes(args)
    print("Number of Road: {}".format(len(roads_dict)))

    node_ids = list(nodes_dict.keys())
    # road_ids = list(roads_dict.keys())

    G = nx.DiGraph(nodetype=int)

    # add nodes to the graph
    G.add_nodes_from(node_ids)

    # add edges to the graph with length
    for i in tqdm(range(len(node_ids)), desc="road_graph"):
        u = node_ids[i]
        out_nodes = adj_nodes[u]

        for k in range(len(out_nodes)):
            len_k = find_length_by_nodes(roads_dict, u, out_nodes[k])
            G.add_edge(u, out_nodes[k], length=len_k)

    write_path = args.workspace + 'road_graph_wlen.pkl'
    if os.path.exists(write_path):
        print("File road_graph_wlen.pkl exists. Will be overwritten.")
    else:
        with open(write_path, 'wb') as f:
            pickle.dump(G, f)
            print("Saved road_graph_wlen.pkl.")


def print_graph(args):
    graph_path = args.workspace + 'road_graph_wlen.pkl'

    with open(graph_path, 'rb') as f:
        G = pickle.load(f)
    print("Loaded road_graph_wlen.pkl.")

    # print graph
    graph_nodes = G.nodes()
    graph_edges = G.edges(data=True)

    print("Nodes Num:", len(graph_nodes))

    # for edge in graph_edges:
    #     if len(edge) == 3:
    #         source, target, data = edge
    #         if "length" in data:
    #             length = data["length"]
    #             print(f"Edge: {source} -> {target}, Length: {length}")
    #         else:
    #             print(f"Edge: {source} -> {target}, No length data available")
    #     elif len(edge) == 2:
    #         source, target = edge
    #         print(f"Edge: {source} -> {target}, No data available")


def gen_node2vec_emb(args):
    edges_weight_file = "traj_freq.txt"  # edge frequency
    G = nx.read_edgelist(os.path.join(args.workspace, edges_weight_file), nodetype=int, create_using=nx.DiGraph(), data=[('weight', int)])
    print("Nodes (in graph): {}, Edges (in graph): {}".format(len(G.nodes), len(G.edges)))

    node_num = args.node_num
    print("Node Number: {}".format(node_num))

    model = Node2Vec(G, walk_length=args.walk_len, num_walks=args.num_walks, p=args.p, q=args.q, workers=32, use_rejection_sampling=0)

    start_time = time.time()
    model.train(embed_size=args.emb_dim, window_size=args.window, epochs=args.epoch)
    end_time = time.time()
    print("Training Time:" + '{:.3f}s'.format(end_time-start_time))

    embeddings = model.get_embeddings()  # dict
    print("Expected: ({}, {})".format(node_num, args.emb_dim))

    weights = []

    for i in range(node_num):
        if i in embeddings:
            # print("Both out_degree and in_degree for Node {} are not 0".format(i))
            weights.append(embeddings[i])
        else:
            # print("Both out_degree and in_degree for Node {} are 0".format(i))
            weights.append(np.random.normal(0, 1, size=args.emb_dim))

    node_embedding = np.array(weights, dtype=np.float32)
    print("node_embedding.shape:", node_embedding.shape)

    embedding_file = "nodes_embedding_{}.npy".format(node_embedding.shape[1])
    embedding_path = os.path.join(args.workspace, embedding_file)
    np.save(embedding_path, node_embedding)

    print("Node2Vec Embedding Saved Successful.")

