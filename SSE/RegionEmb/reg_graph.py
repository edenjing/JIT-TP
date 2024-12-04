#!/usr/bin/env python
# -*- coding: UTF-8 -*-


import argparse
import networkx as nx
from tqdm import tqdm
import pickle
import os
import numpy as np
from NodeEmb.ge.models.node2vec import Node2Vec
import time
import json


def read_att_matrix(args):
    fp = os.path.join(args.workspace, "30/query_info/att_matrix.txt")
    att_matrix = np.loadtxt(fp, dtype=int)
    att_matrix = att_matrix.tolist()
    print(f"Read att_matrix finished.")
    return att_matrix


def read_adj_region(args):
    fp = os.path.join(args.workspace, "30/adj_grids_30.json")
    with open(fp, 'r') as f:
        adj_ids = json.load(f)
    print(f"Read adj_grids_30.json finished.")
    return adj_ids


def get_region_graph(args, adj_nodes, att_matrix):
    node_ids = [i for i in range(args.region_num)]

    # write the Edgelist with weight
    write_path = args.workspace + '30/region_edges_30.txt'
    with open(write_path, 'w') as f:
        for i in range(len(node_ids)):
            u = node_ids[i]
            out_nodes = adj_nodes[str(u)]

            for k in range(len(out_nodes)):
                len_k = att_matrix[u][out_nodes[k]]
                if len_k != 0:
                    line = ' '.join(map(str, [u, out_nodes[k], len_k]))
                    f.write(line + "\n")
    print("Write region_edges_30.txt finished.")

    # G = nx.DiGraph(nodetype=int)
    # # add nodes to the graph
    # G.add_nodes_from(node_ids)
    # # add edges to the graph with length
    # for i in tqdm(range(len(node_ids)), desc="road_graph"):
    #     u = node_ids[i]
    #     out_nodes = adj_nodes[str(u)]
    #
    #     for k in range(len(out_nodes)):
    #         len_k = att_matrix[u][out_nodes[k]]
    #         G.add_edge(u, out_nodes[k], length=len_k)

    # print_graph
    # graph_nodes = G.nodes()
    # graph_edges = G.edges(
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
    edges_weight_file = "30/region_edges_30.txt"
    G = nx.read_edgelist(os.path.join(args.workspace, edges_weight_file), nodetype=int, create_using=nx.DiGraph(), data=[('weight', int)])

    print("Nodes (in graph): {}, Edges (in graph): {}".format(len(G.nodes), len(G.edges)))

    node_num = args.region_num
    print("Node Number: {}".format(node_num))

    model = Node2Vec(G, walk_length=args.walk_len, num_walks=args.num_walks, p=args.p, q=args.q, workers=32, use_rejection_sampling=0)

    start_time = time.time()
    model.train(embed_size=args.emb_dim, window_size=args.window, epochs=args.epochs)
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
    print("region_embedding.shape:", node_embedding.shape)

    embedding_file = "region_embedding_{}.npy".format(node_embedding.shape[1])
    embedding_path = os.path.join(args.workspace, embedding_file)
    np.save(embedding_path, node_embedding)

    print("Region Embedding Saved Successful.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=str, default="/data5/edenjingzhao/QDTP/data/BJData/bj_data/")
    parser.add_argument('--region_num', type=int, default=30*30)
    parser.add_argument('--epochs', type=int, default=10)

    # for node2vec
    parser.add_argument('--walk_len', type=int, default=30)
    parser.add_argument('--num_walks', type=int, default=25)
    parser.add_argument('--p', type=float, default=1.0)
    parser.add_argument('--q', type=float, default=1.0)
    parser.add_argument('--emb_dim', type=int, default=64)
    parser.add_argument('--window', type=int, default=5)

    opt = parser.parse_args()
    print(opt)

    att_matrix = read_att_matrix(opt)
    adj_nodes = read_adj_region(opt)
    get_region_graph(opt, adj_nodes, att_matrix)
    gen_node2vec_emb(opt)



if __name__ == '__main__':
    main()
