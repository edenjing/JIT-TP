#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
Functions: 1. prepare_sse_input(); 2. process_sse_output(); 3. prepare_rtsp_input()
'''
import torch
from typing import Dict, List
import numpy as np
import json


# Mapping node IDs to region IDs using grid information
def node_region_mapping(fp):
    """
    Loads and returns a dictionary mapping node IDs to region IDs.
    Args:
        json_path: Path to the JSON file containing grid information
    Returns:
        Dict[int, int]: Dictionary mapping node IDs to region IDs
    """
    # Load the grids information
    with open(fp, 'r') as f:
        grids_info = json.load(f)

    # Create mapping dictionary
    node_to_region_dict = {}

    # Process the grids information
    for region_id, region_info in grids_info.items():
        region_id = int(region_id)  # Convert string keys to int
        for node_id in region_info['node_ids']:
            node_to_region_dict[node_id] = region_id

    return node_to_region_dict


def prepare_sse_input(query: dict, device: str, region_size: int, node_to_region_dict) -> Dict[str, torch.Tensor]:
    """
    Prepare input for SSE model inference, maintaining same structure as training.
    Args:
        query: Query object containing:
            - origin: origin node
            - destination: destination node
        device: device to place tensors on ('cpu' or 'cuda')
        region_size: total number of regions
        node_to_region_fn: function to convert node ID to region ID
    Returns:
        Dict[str, torch.Tensor]: Prepared input tensors for SSE model
    """
    # Extract query information
    origin = query['origin']
    destination = query['destination']

    # Initialize input tensors
    o = torch.tensor([origin], dtype=torch.long)
    d = torch.tensor([destination], dtype=torch.long)

    # Get origin and destination regions
    o_reg = torch.tensor([node_to_region_dict(origin)], dtype=torch.long)
    d_reg = torch.tensor([node_to_region_dict(destination)], dtype=torch.long)

    # Create dummy label tensor (zeros) with shape [1, region_size]
    label = torch.zeros(1, region_size, dtype=torch.float)

    # Create regions tensor (all possible regions)
    regions = torch.arange(region_size, dtype=torch.long).unsqueeze(0)  # Shape: [1, region_size]

    # Move tensors to correct device
    o = o.to(device)
    d = d.to(device)
    o_reg = o_reg.to(device)
    d_reg = d_reg.to(device)
    label = label.to(device)
    regions = regions.to(device)

    return {
        'origin': o,
        'destination': d,
        'origin_region': o_reg,
        'destination_region': d_reg,
        'label': label,
        'regions': regions
    }


def process_sse_output(outputs: torch.Tensor, gama: float = 0.5) -> List[int]:
    """
    Process SSE model outputs to get predicted region IDs.
    Args:
        outputs: Model output tensor of shape [1, region_size] containing probabilities
        gama: Probability threshold for selecting prediction regions (default: 0.5)
    Returns:
        List[int]: List of predicted region IDs
    """
    # Move outputs to CPU and convert to numpy for processing
    outputs = outputs.detach().cpu().numpy().squeeze()  # Remove batch dimension

    # Get region indices where probability exceeds threshold
    predicted_regions = np.where(outputs >= gama)[0].tolist()

    # If no region exceeds threshold, at least return the region with highest probability
    if not predicted_regions:
        predicted_regions = [int(np.argmax(outputs))]

    return predicted_regions


def prepare_rtsp_input(speed_data: dict, hist_len: int, device: str = 'cuda') -> Dict[
    str, torch.Tensor]:
    """
    Prepare input for RTSP model inference.
    Args:
        speed_data: Dictionary of {region_id: speed_array} containing recent speed data
        regions_edges: Dictionary mapping region_id to list of edge indices
        hist_len: Length of historical data needed
        device: Device to place tensors on ('cpu' or 'cuda')
    Returns:
        Dict[str, Tensor]: Dictionary containing prepared input tensor
    """
    # Get number of regions and max number of edges in any region
    num_regions = len(speed_data)
    max_edges = max(speed_array.shape[0] for speed_array in speed_data.values())

    # Create padded tensor to hold all regions' data
    # Shape: [batch_size=1, num_regions, max_edges, hist_len]
    padded_hist = torch.zeros(1, num_regions, max_edges, hist_len)

    # Create mask for valid edges
    # Shape: [batch_size=1, num_regions, max_edges]
    valid_edges_mask = torch.zeros(1, num_regions, max_edges)

    for idx, (region_id, speed_array) in enumerate(speed_data.items()):
        # Convert speed data to tensor if it's not already
        if not isinstance(speed_array, torch.Tensor):
            speed_array = torch.FloatTensor(speed_array)

        # Get the most recent hist_len timestamps of data
        if speed_array.shape[1] < hist_len:
            raise ValueError(f"Not enough historical data. Need {hist_len} timestamps, but got {speed_array.shape[1]}")

        num_edges = speed_array.shape[0]
        recent_data = speed_array[:, -hist_len:]  # [num_edges_in_region, hist_len]

        # Fill in the padded tensor
        padded_hist[0, idx, :num_edges, :] = recent_data

        # Mark valid edges in the mask
        valid_edges_mask[0, idx, :num_edges] = 1

    return {
        'region_hist': padded_hist.to(device),
        'valid_edges_mask': valid_edges_mask.to(device)
    }