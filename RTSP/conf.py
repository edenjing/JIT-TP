#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# conf.py

class RTSPConfig:
    def __init__(self, opt, global_graph):
        self.model_config = {
            'global_graph': global_graph,
            'time_slot_size': opt.time_slot_size,
            'in_dim': opt.in_dim,
            'spatial_feature_dim': opt.spatial_feature_dim,
            'out_spatial_dim': opt.spatial_feature_dim,
            'out_temporal_dim': opt.temporal_feature_dim,
            'graph_layers': opt.graph_layer,
            'rnn_layers': opt.rnn_layer,
            'spatial_context_dim': opt.spatial_feature_dim,
            'temporal_context_dim': opt.temporal_feature_dim,
            'region_nums': opt.region_nums,
            'edge_nums': opt.edge_nums,
            'hidden_size': opt.hidden_size,
            'pred_len': opt.pred_len,
            'spatial_name_str': 'spatial_feature'
        }

    def get_config(self):
        return self.model_config

    @staticmethod
    def get_default_config():
        """Return default configuration when opt and global_graph not available"""
        return {
            'global_graph': None,
            'time_slot_size': 288,
            'in_dim': 64,
            'spatial_feature_dim': 64,
            'out_spatial_dim': 64,
            'out_temporal_dim': 64,
            'graph_layers': 1,
            'rnn_layers': 1,
            'spatial_context_dim': 64,
            'temporal_context_dim': 64,
            'region_nums': 100,
            'edge_nums': 651748,
            'hidden_size': 64,
            'pred_len': 3,
            'spatial_name_str': 'spatial_feature'
        }