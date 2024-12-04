#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：RTSP 
@File    ：model.py
@Author  ：Eden
@Date    ：2024/11/4 1:31 PM 
'''
import torch
import torch.nn as nn
import dgl
import torch.nn.functional as F
import dgl.nn.pytorch as dglnn
from torch.autograd import Variable


# Goal: generate the global spatio-temporal representations
# spatial feature: region-connection graph
# temporal feature: time interval
class GlobalEncoder(nn.Module):
    def __init__(self, global_graph, in_temporal_dim,
                 in_spatial_dim,
                 out_temporal_dim,
                 out_spatial_dim,
                 rnn_layers,
                 graph_layers,
                 spatial_name_str,
                 device) -> None:
        super(GlobalEncoder, self).__init__()

        self.device = device
        self.num_regions = global_graph.number_of_nodes()

        # Initial embedding layer for regions
        self.region_embedding = nn.Parameter(torch.randn(self.num_regions, in_spatial_dim))

        # Spatial encoding layers (GCN)
        self.spatial_conv = nn.ModuleList([
            dglnn.GraphConv(
                in_spatial_dim if i == 0 else out_spatial_dim,
                out_spatial_dim,
                norm='both',
                weight=True,
                bias=True,
                allow_zero_in_degree=True
            ).to(device) for i in range(graph_layers)
        ])

        # Temporal encoding layer (LSTM)
        self.temporal_rnn = nn.LSTM(
            input_size=1,  # Since time_slots are just indices
            hidden_size=out_temporal_dim,
            num_layers=rnn_layers,
            batch_first=True
        )

        # Additional layers for generating new adjacency matrices
        self.adj_generator = nn.Linear(out_temporal_dim + out_spatial_dim, 1)
        self.spatial_name_str = spatial_name_str
        self.gelu = nn.GELU()

    # Add:
    # 1. the validity weight of adjacent regions
    # 2. the learnable parameter matrix to capture spatial correlation
    def forward(self, global_graph, time_slots): # global_graph, batch_local_g, speeds_h
        # Move graph to the same device as the model
        global_graph = global_graph.to(self.device)

        # 1. Initial region embeddings and spatial encoding
        h_spatial = self.region_embedding  # [num_regions, in_spatial_dim]

        # 2. Encode spatial information using GCN
        for conv in self.spatial_conv:
            h_spatial = self.gelu(conv(global_graph, h_spatial))
        H_s = h_spatial  # [num_regions, out_spatial_dim]

        # 3. Create and encode temporal sequence
        time_indices = torch.arange(time_slots, dtype=torch.float32, device=self.device).unsqueeze(-1)  # [288, 1]
        time_indices = time_indices.unsqueeze(0)  # [1, 288, 1]

        with torch.backends.cudnn.flags(enabled=False):
            H_t, _ = self.temporal_rnn(time_indices)  # [1, 288, out_temporal_dim]
        H_t = H_t.squeeze(0)  # [288, out_temporal_dim]

        # 4. Generate new adjacency matrices for each time interval
        new_spatial_features = []
        for t in range(time_slots):
            # Combine spatial and temporal features
            h_t_current = H_t[t].unsqueeze(0).expand(self.num_regions, -1)  # [num_regions, out_temporal_dim]
            combined_features = torch.cat([H_s, h_t_current], dim=-1)  # [num_regions, out_spatial_dim + out_temporal_dim]

            # Generate adjacency weights
            adj_weights = self.adj_generator(combined_features)  # [num_regions, 1]
            adj_weights = torch.sigmoid(adj_weights)

            # Apply new adjacency weights to spatial features
            weighted_h_s = H_s * adj_weights

            # Additional GCN layer with new adjacency matrix
            for conv in self.spatial_conv:
                weighted_h_s = self.gelu(conv(global_graph, weighted_h_s))

            new_spatial_features.append(weighted_h_s)

        # 5. Average pooling over all time slots
        new_spatial_features = torch.stack(new_spatial_features, dim=0)  # [288, num_regions, out_spatial_dim]
        H_s_final = torch.mean(new_spatial_features, dim=0)  # [num_regions, out_spatial_dim]
        H_t_final = torch.mean(H_t, dim=0)  # [out_temporal_dim]

        # print("H_s Shape: {}".format(H_s_final), "H_t Shape: {}".format(H_t_final))
        return H_s_final, H_t_final


# class Decoder(nn.Module):
#     def __init__(self, spatial_dim, temporal_dim, hidden_dim, output_dim):
#         super().__init__()
#
#         # Dimension normalization layers
#         self.spatial_proj = nn.Linear(spatial_dim + spatial_context_dim, hidden_dim)
#         self.temporal_proj = nn.Linear(temporal_dim + temporal_context_dim, hidden_dim)
#
#         # Spatial processing
#         self.spatial_conv = nn.ModuleList([
#             GraphConv(hidden_dim, hidden_dim),
#             GraphConv(hidden_dim, hidden_dim)
#         ])
#
#         # Temporal processing
#         self.temporal_rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
#
#         # Feature fusion
#         self.fusion = nn.Sequential(
#             nn.Linear(hidden_dim * 2, hidden_dim),
#             nn.GELU(),
#             nn.Linear(hidden_dim, output_dim)
#         )
#
#         self.gelu = nn.GELU()
#         self.norm = nn.LayerNorm(hidden_dim)
#
#     def forward(self, g, hs, ht, c_hs, c_ht):
#         batch_size, seq_len, _ = hs.shape
#
#         # Project concatenated features to same dimension
#         hs = self.temporal_proj(torch.concat([hs, c_ht], dim=-1))
#         ht = self.spatial_proj(torch.concat([ht, c_hs], dim=-1))
#
#         # Spatial path with residual connections
#         spatial_out = ht
#         for conv in self.spatial_conv:
#             spatial_out = spatial_out + self.gelu(conv(g, spatial_out))
#         spatial_out = self.norm(spatial_out)
#
#         # Temporal path
#         with torch.backends.cudnn.flags(enabled=False):
#             temporal_out, _ = self.temporal_rnn(hs)
#         temporal_out = self.norm(temporal_out)
#
#         # Combine features and project to output
#         # Expand spatial features to match temporal sequence length
#         spatial_out = spatial_out.unsqueeze(1).expand(-1, seq_len, -1)
#         combined = torch.cat([temporal_out, spatial_out], dim=-1)
#         output = self.fusion(combined)
#
#         return output

class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim_q, input_dim_kv, output_dim, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads

        # 注意力参数
        self.W_q = nn.Linear(input_dim_q, output_dim)
        self.W_k = nn.Linear(input_dim_kv, output_dim)
        self.W_v = nn.Linear(input_dim_kv, output_dim)
        self.W_o = nn.Linear(output_dim, output_dim)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):
        q = self.W_q(q)  # 查询向量
        k = self.W_k(k)  # 键向量
        v = self.W_v(v)  # 值向量

        # 按照头数进行切分和重塑
        batch_size = q.size(0)
        q = q.view(batch_size, -1, self.num_heads, q.size(-1) // self.num_heads).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, k.size(-1) // self.num_heads).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, v.size(-1) // self.num_heads).transpose(1, 2)

        # print(q.shape, k.shape, v.shape)
        scores = torch.matmul(q, k.transpose(-2, -1))  # 计算注意力分数
        attention_weights = self.softmax(scores)  # 对分数进行softmax归一化

        output = torch.matmul(attention_weights, v)  # 加权求和得到输出
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, output.size(-1) * self.num_heads)  # 重塑输出
        output = self.W_o(output)  # 线性映射得到最终输出

        return output


# Goal: generate the local spatio-temporal representation of individual region
# Note that the global representations are also used, to fuse with local representations.
class SpatioTemporalEncoder(nn.Module):
    def __init__(self, global_graph, time_slot_size, in_temporal_dim, in_spatial_dim, spatial_context_dim, temporal_context_dim,
                hidden_size, out_temporal_dim, out_spatial_dim, rnn_layers, graph_layers,
                region_nums, edge_nums, device, use_global=True, use_local=True, use_fusion=True) -> None:
        super(SpatioTemporalEncoder, self).__init__()
        self.encoder = GlobalEncoder(global_graph,
                                     in_temporal_dim,
                                     in_spatial_dim,
                                     out_temporal_dim, out_spatial_dim, rnn_layers, graph_layers, 'spatial_name_str', device)
        self.time_slot_size = time_slot_size
        self.use_global = use_global
        self.use_local = use_local
        self.use_fusion = use_fusion
        self.out_temporal_dim=out_temporal_dim

        # Context transformation layers
        self.s2ctx = nn.Sequential(
            nn.Linear(out_temporal_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, spatial_context_dim)
        )
        self.t2ctx = nn.Sequential(
            nn.Linear(out_spatial_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, temporal_context_dim)
        )
        self.ctx2t = nn.Sequential(
            nn.Linear(spatial_context_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, out_temporal_dim)
        )
        self.ctx2s = nn.Sequential(
            nn.Linear(temporal_context_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, out_spatial_dim)
        )
        # Local temporal encoder
        self.temporal_rnn = nn.LSTM(input_size=1,
                                    hidden_size=temporal_context_dim,
                                    num_layers=rnn_layers,
                                    batch_first=True)
        # Embeddings and GCN layers
        self.region_embedding = nn.Embedding(region_nums, spatial_context_dim)
        # self.edge_embedding = nn.Embedding(edge_nums, spatial_context_dim)

        self.spatial_proj = nn.Linear(in_features=in_spatial_dim,
                                      out_features=out_spatial_dim)

        self.global_gcn = dglnn.GraphConv(spatial_context_dim, spatial_context_dim, norm='both', weight=True, bias=True, allow_zero_in_degree=True).to(device)
        self.local_gcn = dglnn.GraphConv(
            in_feats=spatial_context_dim,  # Now matches the projected dimension
            out_feats=spatial_context_dim,
            norm='none',
            weight=True,
            bias=True
        )

        # Extension layers
        self.extend_spatial = nn.Sequential(
            nn.Linear(spatial_context_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, temporal_context_dim * time_slot_size)
        )
        self.extend_temporal = nn.Sequential(
            nn.Linear(spatial_context_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, temporal_context_dim)
        )

        self.extend_beta = nn.Sequential(
            nn.Linear(out_temporal_dim + out_spatial_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, out_temporal_dim+out_spatial_dim)
        )

        # Fusion layers
        self.spc2tmo = nn.Linear(spatial_context_dim, out_temporal_dim)
        self.spc2spo = nn.Linear(spatial_context_dim, out_spatial_dim)
        self.spo2tmo = nn.Linear(out_spatial_dim, out_temporal_dim)
        self.tmo2spo = nn.Linear(out_temporal_dim, out_spatial_dim)

        # Attention Mechanism
        self.s2t_attention = MultiHeadAttention(out_temporal_dim, out_spatial_dim,
                                               out_spatial_dim + out_temporal_dim, 1)
        self.t2s_attention = MultiHeadAttention(out_spatial_dim, out_temporal_dim,
                                              out_temporal_dim + out_spatial_dim, 1)

        # Final fusion layer
        self.fusion = nn.Sequential(
            nn.Linear((out_spatial_dim + out_temporal_dim)*2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, (out_spatial_dim + out_temporal_dim)*2)
        )

        # Store dimensions
        self.spatial_context_dim = spatial_context_dim
        self.temporal_context_dim = temporal_context_dim
        self.out_spatial_dim = out_spatial_dim

        # Initialize global graph
        # self.global_g = dgl.graph(region_edges, num_nodes=region_nums).to(device)
        # self.global_regions = torch.arange(region_nums).to(device)

        self.gelu = nn.GELU()

    def forward(self, global_graph, time_slots, global_spatial_idx, batch_local_g, speeds_h,
                local_batch_idx, local_spatial_idx, local_spatial_feature):
        """
        Args:
            global_graph: DGL graph for global region connectivity
            time_slots: Time interval indices
            global_spatial_idx: Global region indices
            batch_local_g: DGL graph for local region connectivity
            speeds_h: Speed profiles [batch_size, edge_size, hist_len]
            local_batch_idx: Batch indices for local features
            local_spatial_idx: Local region indices
            local_spatial_feature: Local spatial features
        """
        # Get global representations
        hs, ht = self.encoder(global_graph, time_slots)

        # 1. Local Temporal Encoding
        bs, edge_size, hist_len = speeds_h.shape

        # Reshape speeds_h to [bs * edge_size, hist_len, in_temporal_dim]
        temporal_input = speeds_h.reshape(-1, hist_len, 1)

        with torch.backends.cudnn.flags(enabled=False):
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            # Process through LSTM
            local_temporal_h, _ = self.temporal_rnn(temporal_input)  # [bs * edge_size, hist_len, temporal_context_dim]

            # Take the last hidden state and reshape back
            local_temporal_h = local_temporal_h[:, -1, :]  # [bs * edge_size, temporal_context_dim]
            local_temporal_h = local_temporal_h.reshape(bs, edge_size, -1)  # [bs, edge_size, temporal_context_dim]
            # print("local_temporal_h:", local_temporal_h)

        # 2. Local Spatial Encoding
        local_spatial_h = self.spatial_proj(local_spatial_feature)  # Project spatial features
        local_spatial_h = self.gelu(self.local_gcn(batch_local_g, local_spatial_h))  # [local node num, spatial_context_dim]
        # print("local_spatial_h.shape:", local_spatial_h.shape)

        # 3. Spatial Fusion
        # Incorporate global region representations into local representations
        c_hs = hs + local_spatial_h.sum(dim=0).unsqueeze(0)

        # 4. Temporal Fusion
        # Incorporate global temporal representations
        c_ht = local_temporal_h + ht.view(1, 1, -1).expand_as(local_temporal_h)

        # Apply global/local switches
        if not self.use_global:
            c_hs = torch.zeros_like(c_hs)
            c_ht = torch.zeros_like(c_ht)
        if not self.use_local:
            hs = self.ctx2s(c_ht)
            ht = self.ctx2t(c_hs)

        # 5. Generate initial representations for attention
        if self.use_fusion:
            # Transform representations
            spatial_rep = self.spc2spo(c_hs)
            temporal_rep = self.spc2tmo(c_ht)
            ht_expanded = ht.unsqueeze(0).expand(bs, edge_size, -1)
            alpha = spatial_rep + self.spo2tmo(hs)
            beta = temporal_rep + self.tmo2spo(ht_expanded)
        else:
            alpha = self.spo2tmo(hs) + self.spc2tmo(local_temporal_h)
            beta = self.tmo2spo(ht) + self.spc2spo(local_spatial_h)

        # 6. Apply attention mechanism
        alpha_local = alpha.unsqueeze(0).expand(bs, -1, -1)
        beta_local = beta.view(bs, edge_size, beta.shape[-1])

        # out_temporal_dim, out_spatial_dim,
        # out_spatial_dim + out_temporal_dim, 1
        # S2T and T2S attention
        beta_enhanced = self.s2t_attention(beta_local, alpha_local, alpha_local)
        alpha_enhanced = self.t2s_attention(alpha_local, beta_local, beta_local)

        # 7. Final fusion
        beta_temporal = self.extend_beta(beta_enhanced)

        # Combine spatial and temporal representations  edge->target, curr->alpha_enhanced.shape[1]
        if alpha_enhanced.shape[1] <= edge_size:
            padded_alpha_enhanced = torch.zeros((bs, edge_size, self.out_temporal_dim + self.out_spatial_dim),
                                                device=alpha_enhanced.device)
            padded_alpha_enhanced[:, :alpha_enhanced.shape[1], :] = alpha_enhanced
        elif alpha_enhanced.shape[1] > edge_size:
            padded_alpha_enhanced = alpha_enhanced[:, :edge_size, :]

        # padded_alpha_enhanced = torch.zeros((bs, edge_size, self.out_temporal_dim+self.out_spatial_dim), device=alpha_enhanced.device)
        # padded_alpha_enhanced[:, :alpha_enhanced.shape[1], :] = alpha_enhanced

        # print("alpha_enhanced.shape:", alpha_enhanced.shape)

        combined = torch.cat([padded_alpha_enhanced, beta_temporal], dim=-1)
        final_rep = self.fusion(combined)

        return combined, final_rep


class RegionTSP(nn.Module):
    def __init__(self, global_graph, time_slot_size, in_dim, spatial_feature_dim, out_spatial_dim, out_temporal_dim,
                 graph_layers, rnn_layers, spatial_context_dim, temporal_context_dim, region_nums,
                 edge_nums, hidden_size, pred_len, spatial_name_str='spatial_feature', device='cpu') -> None:
        super(RegionTSP, self).__init__()

        self.spatial_name_str = spatial_name_str
        self.device = device
        self.pred_len = pred_len

        # Main encoder for feature extraction
        self.STencoder = SpatioTemporalEncoder(
            global_graph=global_graph,
            time_slot_size=time_slot_size,
            in_temporal_dim=in_dim,
            in_spatial_dim=in_dim,
            spatial_context_dim=spatial_context_dim,
            temporal_context_dim=temporal_context_dim,
            hidden_size=hidden_size,
            out_temporal_dim=out_temporal_dim,
            out_spatial_dim=out_spatial_dim,
            rnn_layers=rnn_layers,
            graph_layers=graph_layers,
            region_nums=region_nums,
            edge_nums=edge_nums,
            device=device,
            use_global=True,
            use_local=True,
            use_fusion=True
        )

        # Prediction layers
        self.predict_fc = nn.Sequential(
            nn.Linear((out_spatial_dim + out_temporal_dim)*2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, in_dim * pred_len)
        )

        self.reset_parameters()

    # def reset_parameters(self):
    #     """Reinitialize learnable parameters."""
    #     gain = nn.init.calculate_gain('leaky_relu', 0.2)
    #     for name, param in self.STencoder.named_parameters():
    #         if "norm" in name:
    #             nn.init.zeros_(param)
    #         elif "weight" in name:
    #             nn.init.xavier_normal_(param, gain=gain)
    #         else:
    #             nn.init.zeros_(param)
    #
    #     for name, param in self.predict_fc.named_parameters():
    #         if "weight" in name:
    #             nn.init.xavier_normal_(param, gain=gain)
    #         else:
    #             nn.init.zeros_(param)

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('leaky_relu', 0.2)

        # Initialize STencoder parameters
        for name, param in self.STencoder.named_parameters():
            if "norm" in name:
                nn.init.zeros_(param)
            elif "weight" in name:
                if len(param.shape) >= 2:  # Check if parameter has at least 2 dimensions
                    nn.init.xavier_normal_(param, gain=gain)
                else:
                    nn.init.uniform_(param, -0.1, 0.1)
            else:
                nn.init.zeros_(param)

        # Initialize predict_fc parameters
        for name, param in self.predict_fc.named_parameters():
            if "weight" in name:
                if len(param.shape) >= 2:  # Check if parameter has at least 2 dimensions
                    nn.init.xavier_normal_(param, gain=gain)
                else:
                    nn.init.uniform_(param, -0.1, 0.1)
            else:
                nn.init.zeros_(param)


    def forward(self, global_graph, time_slot_size, batch_local_g, traffic_h):
        """
        Forward pass of the RegionTSP model

        Args:
            batch_local_graph: DGL graph containing local connectivity information
            region_hist: Historical traffic speed data [batch_size, num_edges, hist_len]
            global_spatial_idx: Global spatial indices
        Returns:
            predictions: Predicted future speed values [batch_size, num_edges, pred_len]
        """
        local_batch_idx = batch_local_g.ndata['batch_idx']
        local_spatial_idx = batch_local_g.ndata['spatial_idx']
        local_spatial_feature = batch_local_g.ndata[self.spatial_name_str]

        # Get spatio-temporal representations
        raw_embeddings, st_embeddings = self.STencoder(
            global_graph,
            time_slot_size,
            local_spatial_idx,
            batch_local_g,
            traffic_h,
            local_batch_idx,
            local_spatial_idx,
            local_spatial_feature
        )

        # Combine embeddings
        combined_embeddings = raw_embeddings + st_embeddings

        # Generate predictions for future time steps
        predictions = self.predict_fc(combined_embeddings)
        batch_size, num_edges = predictions.shape[0], predictions.shape[1]
        predictions = predictions.view(batch_size, num_edges, -1, self.pred_len)
        pred = predictions.mean(dim=2)
        return pred

    # def calculate_loss(self, predictions, targets, mask=None):
    #     """
    #     Calculate MSE loss with optional masking
    #
    #     Args:
    #         predictions: Predicted speed values [batch_size, num_edges, pred_len]
    #         targets: Ground truth speed values [batch_size, num_edges, pred_len]
    #         mask: Optional mask for valid values
    #     """
    #     if mask is not None:
    #         loss = F.mse_loss(predictions[mask], targets[mask])
    #     else:
    #         loss = F.mse_loss(predictions, targets)
    #     return loss