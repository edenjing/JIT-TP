#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
Generate Edge embeddings.
'''

import torch
import torch.nn as nn
import math
import json
import argparse
import os
import numpy as np


def get_sp_data(sp_path, edge_num):
    sp_fp = sp_path + "BJSP.txt"
    SP_Data = {}

    # read only the first line
    with open(sp_fp, 'r') as f:
        for line in f:
            values = line.strip().split()
            edge_id = int(values[0])
            if edge_id not in SP_Data and edge_id < edge_num:
                SP_Data[edge_id] = {}
                for i in range(2, len(values), 2):
                    timestamp = int(values[i])
                    # travel time
                    time = int(values[i+1])
                    SP_Data[edge_id][timestamp] = time
            else:
                continue
    # print(len(SP_Data), "\n", SP_Data[0])
    return SP_Data


class PreTrainedEdgeTemporalEmbedding(nn.Module):
    def __init__(self, num_time_slots, embedding_dim):
        super(PreTrainedEdgeTemporalEmbedding, self).__init__()
        self.num_time_slots = num_time_slots
        self.embedding_dim = embedding_dim
        self.time_encoder = nn.Linear(3, embedding_dim // 2)  # 3 for time, sin, cos
        self.travel_time_encoder = nn.Linear(1, embedding_dim // 2)
        self.lstm = nn.LSTM(embedding_dim, embedding_dim, batch_first=True)
        self.attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=4)
        self.fc = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, edge_data):
        # extract time_slot list and travel_time list
        time_slots = list(edge_data.keys())
        travel_times = list(edge_data.values())

        # convert time slots to tensor
        time_slots = torch.tensor([int(ts) for ts in time_slots]).float().unsqueeze(1)
        # Encode cyclical time patterns
        # cyclical encoding / circular encoding
        sin_time = torch.sin(2 * math.pi * time_slots / self.num_time_slots)
        cos_time = torch.cos(2 * math.pi * time_slots / self.num_time_slots)
        time_features = torch.cat([time_slots, sin_time, cos_time], dim=-1)

        travel_times = torch.tensor(travel_times).float().unsqueeze(1)

        time_embedding = self.time_encoder(time_features)
        travel_time_embedding = self.travel_time_encoder(travel_times)

        combined_embedding = torch.cat([time_embedding, travel_time_embedding], dim=-1)

        # Apply LSTM
        lstm_out, _ = self.lstm(combined_embedding.unsqueeze(0))

        # Apply attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # Global pooling and final projection
        pooled_embedding = torch.mean(attn_out.squeeze(0), dim=0)
        edge_embedding = self.fc(pooled_embedding)

        return edge_embedding

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    @classmethod
    def load_model(cls, path, num_time_slots, embedding_dim):
        model = cls(num_time_slots, embedding_dim)
        model.load_state_dict(torch.load(path))
        model.eval()
        return model


class EdgeEmbeddingGenerator:
    def __init__(self, model_path, num_time_slots, embedding_dim):
        self.model = PreTrainedEdgeTemporalEmbedding.load_model(model_path, num_time_slots, embedding_dim)
        self.embeddings = {}

    def generate_embedding(self, edge_id, edge_data):
        with torch.no_grad():
            embedding = self.model(edge_data).numpy()
        self.embeddings[edge_id] = embedding.tolist()
        return embedding

    def save_embeddings(self, path):
        # with open(path, 'w') as f:
        #     json.dump(self.embeddings, f)
        embedding_file = "edges_embedding_{}.npy".format(self.embeddings.shape[1])
        embedding_path = os.path.join(path, embedding_file)
        np.save(embedding_path, self.embeddings)

    def load_embeddings(self, path):
        embedding_file = "edges_embedding_{}.npy".format(self.embeddings.shape[1])
        embedding_path = os.path.join(path, embedding_file)
        self.embeddings = np.load(embedding_path)

        # with open(path, 'r') as f:
        #     self.embeddings = json.load(f)


def train_model(train_data, num_time_slots, embedding_dim, epochs):
    model = PreTrainedEdgeTemporalEmbedding(num_time_slots, embedding_dim)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.MSELoss()  # Example loss function, adjust as needed

    for epoch in range(epochs):
        total_loss = 0
        print("=========Epoch: {}=========".format(epoch))
        for edge_id, edge_data in train_data.items():
            optimizer.zero_grad()
            output = model(edge_data)
            loss = criterion(output, torch.zeros_like(output))  # Example target, adjust as needed
            print("Loss: {:.4f}".format(loss))
            total_loss += loss
            loss = loss / len(edge_data)
            loss.backward()
            optimizer.step()
        total_loss = round(total_loss / len(train_data), 4)
        print(f"Epoch {epoch + 1}/{epochs} completed, Total Loss: {total_loss}")

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bj_raw_path', type=str, default="/data5/edenjingzhao/QDTP/data/BJData/raw/")
    parser.add_argument('--edge_num', type=int, default=651748)
    parser.add_argument('--num_time_slots', type=int, default=24)
    parser.add_argument('--edge_embed_dim', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument("--save_path", type=str, default="/data5/edenjingzhao/QDTP/data/BJData/bj_data/")

    opt = parser.parse_args()
    print(opt)

    train_data = get_sp_data(opt.bj_raw_path, opt.edge_num)

    # Training
    trained_model = train_model(train_data, opt.num_time_slots, opt.edge_embed_dim, opt.epochs)
    trained_model.save_model("edge_embedding_model.pth")

    # Generating embeddings
    generator = EdgeEmbeddingGenerator("edge_embedding_model.pth", opt.num_time_slots, opt.edge_embed_dim)

    for edge_id, edge_data in train_data.items():
        generator.generate_embedding(edge_id, edge_data)

    generator.save_embeddings(opt.save_path)


if __name__ == '__main__':
    main()

