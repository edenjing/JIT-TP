#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
Generate Region Embeddings.
'''

import torch
import torch.nn as nn
import os
import numpy as np
import json
import argparse


def load_edge_embeddings(edge_embedding_path):
    with open(edge_embedding_path, 'r') as f:
        edge_embeddings = json.load(f)
    return edge_embeddings


class RegionEmbedding(nn.Module):
    def __init__(self, edge_embedding_dim, region_embedding_dim):
        super(RegionEmbedding, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=edge_embedding_dim, num_heads=4)
        self.fc = nn.Linear(edge_embedding_dim, region_embedding_dim)

    def forward(self, edge_embeddings):
        # edge_embeddings: tensor of shape (num_edges, edge_embedding_dim)
        edge_embeddings = edge_embeddings.unsqueeze(1)  # Add sequence length dimension
        attn_out, _ = self.attention(edge_embeddings, edge_embeddings, edge_embeddings)
        pooled = torch.mean(attn_out, dim=0)
        region_embedding = self.fc(pooled)
        return region_embedding.squeeze(0)

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    @classmethod
    def load_model(cls, path, edge_embedding_dim, region_embedding_dim):
        model = cls(edge_embedding_dim, region_embedding_dim)
        model.load_state_dict(torch.load(path))
        model.eval()
        return model


class RegionEmbeddingGenerator:
    def __init__(self, region_embedding_dim, edge_embeddings):
        self.region_embedding_dim = region_embedding_dim
        self.edge_embeddings = edge_embeddings
        self.region_embeddings = {}


    def generate_region_embedding(self, region_id):
        # region_data is a dict of edge_id: edge_data pairs
        # edge_embeddings = []
        # for edge_id, edge_data in region_data.items():
        #     edge_embedding = self.edge_generator.generate_embedding(edge_id, edge_data)
        #     edge_embeddings.append(edge_embedding)

        edge_embeddings_tensor = torch.tensor(self.edge_embeddings)


        self.region_embeddings[region_id] = region_embedding.tolist()
        return region_embedding

    def save_region_embeddings(self, path):
        with open(path, 'w') as f:
            json.dump(self.region_embeddings, f)

    def load_edge_embeddings(self, path):
        embedding_file = "edges_embedding_64.npy"
        embedding_path = os.path.join(path, embedding_file)
        self.embeddings = np.load(embedding_path)

        # with open(path, 'r') as f:
        #     self.embeddings = json.load(f)


    def load_region_embeddings(self, path):
        with open(path, 'r') as f:
            self.region_embeddings = json.load(f)


def train_region_model(train_data, edge_embedding_dim, region_embedding_dim, epochs=10):
    model = RegionEmbedding(edge_embedding_dim, region_embedding_dim)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.MSELoss()  # Example loss function, adjust as needed

    for epoch in range(epochs):
        for region_id, edge_embeddings in train_data.items():
            optimizer.zero_grad()
            edge_embeddings_tensor = torch.tensor(edge_embeddings)
            output = model(edge_embeddings_tensor)
            loss = criterion(output, torch.zeros_like(output))  # Example target, adjust as needed
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{epochs} completed")

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bj_raw_path', type=str, default="/data5/edenjingzhao/QDTP/data/BJData/raw/")
    parser.add_argument("-save_path", type=str, default="/data5/edenjingzhao/QDTP/data/BJData/bj_data/")
    parser.add_argument('--region_num', type=int, default=30*30)
    parser.add_argument('--region_embed_dim', type=int, default=64)
    parser.add_argument('-epochs', type=int, default=2)
    opt = parser.parse_args()
    print(opt)


    region_data = get_data(opt)

    # Create some sample data for region model training
    train_data = {
        "region1": [
            [0.1, 0.2, ..., 0.64],  # edge1 embedding
            [0.2, 0.3, ..., 0.64],  # edge2 embedding
            # ... more edge embeddings
        ],
        "region2": [
            [0.15, 0.25, ..., 0.64],  # edge1 embedding
            [0.22, 0.33, ..., 0.64],  # edge2 embedding
            # ... more edge embeddings
        ],
        # ... more regions
    }

    trained_region_model = train_region_model(train_data, edge_embedding_dim, region_embedding_dim)
    trained_region_model.save_model("region_embedding_model.pth")

    # Generating region embeddings
    generator = RegionEmbeddingGenerator(
        edge_model_path="edge_embedding_model.pth",
        region_model_path="region_embedding_model.pth",
        num_time_slots=24,
        edge_embedding_dim=64,
        region_embedding_dim=128
    )

    # Sample region data
    region_data = {
        "edge1": [(0, 15), (1, 16), (2, 15), ..., (23, 13)],
        "edge2": [(0, 14), (1, 15), (2, 13), ..., (23, 13)],
        # ... more edges in the region
    }

    region_embedding = generator.generate_region_embedding("region1", region_data)
    print("Region Embedding:", region_embedding)

    generator.save_region_embeddings("region_embeddings.json")

    # Later, load region embeddings
    generator.load_region_embeddings("region_embeddings.json")


if __name__ == '__main__':
    main()
