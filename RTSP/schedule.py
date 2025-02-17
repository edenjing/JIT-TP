#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
Task: Implement the scheduling part
1. read query
2. get search space regions by SSE model
3. determine the regions need to be updated
4. compute the prediction priority score for regions
5. perform the prediction process for ordered regions by RTSP model
'''
import torch
from utils import dict_to_object
from sse import SSRegionPred
import numpy as np
from typing import Dict, List, Set, Tuple
from model import RegionTSP
from train_test import load_model   # RTSP model
from conf import RTSPConfig
from preprocess.regionData import get_adjacent_regions
import random
from in_out_data import node_region_mapping, prepare_sse_input, process_sse_output, prepare_rtsp_input


class RegionPredictionScheduler:
    def __init__(self, sse_model_path, rtsp_model_path, spatial_importance_matrix,
                 lifetime_threshold, global_graph=None, device = 'cpu'):
        self.device = device
        self.W_s = spatial_importance_matrix
        self.t_s = lifetime_threshold  # e.g., 300: 5 minutes threshold

        # load SSE model
        self.sse_model, self.sse_hparams = load_sse_model(sse_model_path, device)
        self.sse_model.eval()

        # Get config and initialize RTSP model
        if global_graph is None:
            self.rtsp_config = RTSPConfig.get_default_config()
        else:
            self.rtsp_config = RTSPConfig(opt=None, global_graph=global_graph).get_config()

        self.rtsp_model = RegionTSP(**self.rtsp_config, device=device).to(device)
        self.rtsp_model, self.rtsp_metrics = load_model(self.rtsp_model, rtsp_model_path, device)
        self.rtsp_model.eval()

        self.validity_calculator = ValidityWeightCalculator(self.t_s)

    def compute_priority_score(self, region_id, adjacent_regions, current_time):
        """
        Compute priority score for a region based on spatial importance and time validity.
        Args:
            region: Region object
            current_time: Current timestamp
            adjacent_regions: List of adjacent regions
        Returns:
            float: Priority score
        """
        # Get spatial importance weight for the region
        w_s = self.W_s[region_id]

        # Get time validity weight
        validity_weights = self.validity_calculator.compute_validity_weights(region_id, adjacent_regions, current_time)

        # Compute priority score as product of spatial importance and validity weights
        priority_score = w_s * validity_weights[region_id]
        return priority_score


    @torch.no_grad()
    def predict_regions(self, query, current_time: float, fp: str, gridGra :int,
                        node_to_region_dict, gama: float, obv_len: int, device) -> Dict[int, np.ndarray]:
        """
        Predict speed profiles for regions based on query and current time.
        Args:
            query: Query object containing search parameters
            current_time: Current timestamp
            fp: File path to the directory containing adjacency information
            gridGra: Grid granularity parameter
        Returns:
            Dict[int, np.ndarray]: Dictionary mapping region IDs to their updated profiles
        """
        # Get search space regions using SSE model
        sse_input = prepare_sse_input(query, self.device, gridGra*gridGra, node_to_region_dict)
        sse_output = self.sse_model(sse_input)
        search_space_regions = process_sse_output(sse_output, gama)

        # local adjacent regions information
        self.region_adjacency = get_adjacent_regions(fp, gridGra)

        # Compute priority scores for each region
        region_scores = []
        for region_id, region_data in search_space_regions.items():
            # Check if region needs update
            if region_id not in self.validity_calculator.last_update_times:
                self.validity_calculator.last_update_times[region_id] = current_time - self.t_s

            delta_t = current_time - self.validity_calculator.last_update_times[region_id]

            # Get adjacent regions from loaded data
            # adjacent_regions = set(self.region_adjacency.get(region_id, []))

            # compute priority score using actual adjacent regions
            priority_score = self.compute_priority_score(region_id=region_id, current_time=current_time)
            region_scores.append((region_id, region_data, delta_t, priority_score))

        # Sort regions by priority score
        sorted_regions = sorted(region_scores, key=lambda x: x[3], reverse=True)

        # Update profiles for regions that exceed threshold
        updated_profiles = {}
        for region_id, region_data, delta_t, _ in sorted_regions:
            if delta_t >= self.t_s:
                # Prepare and run RTSP prediction
                rtsp_inputs = prepare_rtsp_input(region_data, obv_len, device)
                new_profile = self.rtsp_model(*rtsp_inputs)

                # Store results and update timestamp
                updated_profiles[region_id] = new_profile.cpu().numpy()
                self.validity_calculator.update_timestamp(region_id, current_time)

        return updated_profiles


class ValidityWeightCalculator:
        def __init__(self, t_s: float):
            self.t_s = t_s  # traffic update frequency threshold
            self.last_update_times = {}  # Store last update times for all regions

        def update_timestamp(self, region_id: int, current_time: float):
            """
            Update the last prediction timestamp for a region
            Args:
                region_id: ID of the region that just completed RTSP prediction
                current_time: current timestamp
            """
            self.last_update_times[region_id] = current_time

        def compute_s_metric(self,
                             current_time: float,
                             region_last_update: float) -> float:
            """
            Compute the negative number of prediction cycles since last update
            Args:
                current_time: current timestamp
                region_last_update: region's last update timestamp
            Returns:
                s_metric: negative number of prediction cycles
            """
            return (region_last_update - current_time) / self.t_s

        def compute_validity_weights(self,
                                     target_region_id: int,
                                     adjacent_regions: Set[int],
                                     current_time: float) -> np.ndarray:
            """
            Compute validity weights using stored last update times
            Args:
                target_region_id: ID of the target region
                adjacent_regions: set of adjacent region IDs
                current_time: current timestamp
            Returns:
                validity_weights: array of validity weights
            """
            # Compute s metrics for adjacent regions
            s_metrics = {}
            for adj_region_id in adjacent_regions:
                s_metric = self.compute_s_metric(
                    current_time,
                    self.last_update_times[adj_region_id]
                )
                s_metrics[adj_region_id] = s_metric

            # Compute reciprocal normalization
            reciprocals = {r_id: 1 / abs(s) for r_id, s in s_metrics.items()}
            normalization_factor = sum(reciprocals.values())

            # Calculate normalized weights
            weights = {r_id: rec / normalization_factor
                       for r_id, rec in reciprocals.items()}

            # Set weight to 1 for non-adjacent regions
            all_region_ids = set(self.last_update_times.keys())
            for region_id in all_region_ids:
                if region_id not in adjacent_regions:
                    weights[region_id] = 1.0

            # Convert to ordered array
            n_regions = len(self.last_update_times)
            validity_weights = np.ones(n_regions)
            for region_id, weight in weights.items():
                validity_weights[region_id] = weight

            return validity_weights


def read_query(fp):
    query_path = fp + "/queries.txt"
    queries = []

    with open(query_path, 'r') as f:
        for line in f:
            ori, dst = map(int, line.strip().split())
            queries.append((ori, dst))
    return queries


def generate_timed_queries(queries: List[Tuple[int, int]], start_timestamp: float, time_duration: float) -> List[Dict]:
    """
    Generate timed queries with random departure times within the time domain.
    Args:
        queries: List of (origin, destination) pairs
        start_timestamp: Start time in seconds
        time_duration: Duration in seconds
    Returns:
        List[Dict]: List of queries with timestamps
    """
    timed_queries = []
    for ori, dst in queries:
        # Generate random departure time within the time domain
        departure_time = start_timestamp + random.uniform(0, time_duration)

        query = {
            'timestamp': departure_time,
            'origin': ori,
            'destination': dst,
        }
        timed_queries.append(query)

    # Sort queries by timestamp
    timed_queries.sort(key=lambda x: x['timestamp'])
    return timed_queries


def process_timed_queries(scheduler: RegionPredictionScheduler,
                          timed_queries: List[Dict],
                          fp: str,
                          gridGra: int,
                          node_to_region_dict: Dict[int, int],
                          gama: float,
                          obv_len: int,
                          device: str,
                          time_step: float = 12) -> Dict[float, Dict[int, np.ndarray]]:
    """
    Process queries in temporal order using RegionPredictionScheduler.
    Args:
        scheduler: Initialized RegionPredictionScheduler
        timed_queries: List of queries with timestamps
        fp: File path for adjacency information
        gridGra: Grid granularity
        time_step: Time step for predictions (default: 5 minutes)
    Returns:
        Dict[float, Dict[int, np.ndarray]]: Predictions indexed by timestamp
    """
    all_predictions = {}
    current_time = timed_queries[0]['timestamp']  # Start from first query
    end_time = timed_queries[-1]['timestamp']

    # Process queries in time windows
    while current_time <= end_time:
        # Get queries within the current time window
        window_queries = [
            q for q in timed_queries
            if current_time <= q['timestamp'] < current_time + time_step
        ]

        if window_queries:
            # Combine queries in the same time window
            combined_query = {
                'timestamp': current_time,
                'queries': window_queries,
            }

            # Make predictions
            predictions = scheduler.predict_regions(
                query=combined_query,
                current_time=current_time,
                fp=fp,
                gridGra=gridGra,
                node_to_region_dict=node_to_region_dict,
                gama=gama,
                obv_len=obv_len,
                device=device
            )
            all_predictions[current_time] = predictions

        current_time += time_step

    return all_predictions


def load_sse_model(model_file, device):
    # Load checkpoint with specified device mapping
    checkpoint = torch.load(model_file, map_location=device)

    hparams = dict_to_object(checkpoint['params'])
    hparams.device = device

    model = SSRegionPred(hparams).to(device)

    # Load the trained weights
    model.load_state_dict(checkpoint['model'])
    print('[Info] Trained model state loaded.')
    return model, hparams


def initialize_rtsp_model(rtsp_config, device):
    """Initialize RTSP model with given configuration"""
    rtsp_model = RegionTSP(
        global_graph=rtsp_config['global_graph'],
        time_slot_size=rtsp_config['time_slot_size'],
        in_dim=rtsp_config['in_dim'],
        spatial_feature_dim=rtsp_config['spatial_feature_dim'],
        out_spatial_dim=rtsp_config['out_spatial_dim'],
        out_temporal_dim=rtsp_config['out_temporal_dim'],
        graph_layers=rtsp_config['graph_layers'],
        rnn_layers=rtsp_config['rnn_layers'],
        spatial_context_dim=rtsp_config['spatial_context_dim'],
        temporal_context_dim=rtsp_config['temporal_context_dim'],
        region_nums=rtsp_config['region_nums'],
        edge_nums=rtsp_config['edge_nums'],
        hidden_size=rtsp_config['hidden_size'],
        pred_len=rtsp_config['pred_len'],
        spatial_name_str=rtsp_config.get('spatial_name_str', 'spatial_feature'),
        device=device
    ).to(device)
    return rtsp_model


def main():
    gridGra = 10
    sp_path = "/data5/edenjingzhao/RTSP/data/Beijing/{}/".format(gridGra)
    sse_model_path = "/data5/edenjingzhao/QDTP/data/BJData/bj_model/"
    RTSP_model_path = "/data5/edenjingzhao/RTSP/data/Beijing/{}/model/".format(gridGra)
    adj_region_path = "/home/edenjingzhao/RegionData/".format(gridGra)
    query_path = "/data5/edenjingzhao/QDTP/data/BJData/bj_data/{}/query_info/queries.txt".format(gridGra)

    # define time domain (in seconds)
    time_duration = 48 * 3600  # 48 hours in seconds
    start_timestamp = 1645660800  # Example: Feb 24, 2022, 00:00:00 UTC

    # Read queries
    queries = read_query(query_path)
    # get region-node mapping
    grid_info_path = sp_path + "grid_info.txt"
    node_to_region_dict = node_region_mapping(grid_info_path)

    # Generate timed queries
    timed_queries = generate_timed_queries(queries=queries, start_timestamp=start_timestamp, time_duration=time_duration)

    # Initialize scheduler
    n_regions = gridGra * gridGra
    spatial_importance_matrix = np.ones(n_regions)  # create spatial importance matrix
    lifetime_threshold = 12

    scheduler = RegionPredictionScheduler(
        sse_model_path=sse_model_path,
        rtsp_model_path=RTSP_model_path,
        spatial_importance_matrix=spatial_importance_matrix,
        lifetime_threshold=lifetime_threshold
    )

    # Process queries and get predictions
    predictions = process_timed_queries(
        scheduler=scheduler,
        timed_queries=timed_queries,
        fp=adj_region_path,
        gridGra=gridGra,
        node_to_region_dict=node_to_region_dict,
        gama=0.5,
        obv_len=120,
        device='cuda'if torch.cuda.is_available() else 'cpu',
        time_step=12  # 5 minutes
    )

    print(f"\nQuery Processing Results:")
    print(f"Time domain: {time_duration / 3600:.1f} hours")
    print(f"Total number of queries: {len(timed_queries)}")

    # Print some detailed statistics
    for timestamp, pred_dict in list(predictions.items())[:5]:  # Show first 5 windows
        relative_hour = (timestamp - start_timestamp) / 3600
        print(f"\nPrediction window at {relative_hour:.1f} hours:")
        print(f"Number of updated regions: {len(pred_dict)}")
        if pred_dict:
            regions = list(pred_dict.keys())
            print(f"Updated regions (first 5): {regions[:5]}")

            # Calculate average speeds
            avg_speeds = {
                region_id: np.mean(profile)
                for region_id, profile in pred_dict.items()
            }
            print(f"Average speeds (first 5 regions):")
            for region_id in regions[:5]:
                print(f"  Region {region_id}: {avg_speeds[region_id]:.2f}")


if __name__ == '__main__':
    main()