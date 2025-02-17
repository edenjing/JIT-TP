#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：RTSP
@File    ：train_test.py
@Author  ：Eden
@Date    ：2024/11/4 1:32 PM
'''
import torch
import torch.nn as nn
import argparse
import os
import time
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
import sys
import pickle
import random
import numpy as np
from utils import dict_to_object
from preprocess.speedData import load_data, TrafficDataset
from model import RegionTSP
import psutil
import dgl
from utils import normalize_ground_truth, monitor_memory, stop_monitoring
from conf import RTSPConfig


import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
torch.cuda.empty_cache()


def check_gradients(model):
    """Helper function to check gradient health"""
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                print(f"Bad gradients in {name}")
                return False
            if param.grad.abs().max() > 1000:
                print(f"Large gradients in {name}: {param.grad.abs().max()}")
    return True


def create_global_graph(region_matrix):
    """
    Create a global graph for all regions where:
    - Each node represents one of the 100 regions
    - Edges represent region connections

    Returns:
        g: DGL graph with 100 nodes
    """
    # Convert to numpy if tensor
    if torch.is_tensor(region_matrix):
        region_matrix = region_matrix.numpy()

    # Get edge indices where value is 1
    src_nodes, dst_nodes = np.where(region_matrix == 1)

    # Create DGL graph
    g = dgl.graph((torch.tensor(src_nodes), torch.tensor(dst_nodes)))

    return g


def get_spatial_indices(region_id, region_matrix):
    """
    Orders regions based on connectivity: first the target region,
    then direct neighbors, then remaining regions
    Args:
        region_id: target region ID (int)
        region_matrix: adjacency matrix [num_regions, num_regions]
    Returns:
        spatial_indices: tensor of indices
    """
    # Ensure region_matrix is numpy array
    if not isinstance(region_matrix, np.ndarray):
        region_matrix = np.array(region_matrix)

    num_regions = region_matrix.shape[0]

    # Get directly connected neighbors
    neighbors = np.where(region_matrix[region_id] > 0)[0]

    # Get remaining regions
    all_regions = set(range(num_regions))
    remaining = list(all_regions - set([region_id]) - set(neighbors.tolist()))
    # Order: [target_region, neighbors, remaining]
    ordered_indices = torch.tensor([region_id] + neighbors.tolist() + remaining, dtype=torch.long)

    return ordered_indices


def create_local_graph(region_id, region_matrix,):
    """
    Creates local graph as a DGL graph object
    Args:
        region_id: target region ID (int)
        region_matrix: adjacency matrix [num_regions, num_regions] numpy array
    Returns:
        local_graph: DGL graph with required node features
    """

    if not isinstance(region_matrix, np.ndarray):
        region_matrix = np.array(region_matrix)

    # Get neighbors
    neighbors = np.where(region_matrix[region_id] > 0)[0]

    # Create node mapping: original ID -> local ID
    node_mapping = {region_id: 0}  # target node is always 0
    for i, nid in enumerate(neighbors, 1):
        node_mapping[nid] = i

    # Create edges based on actual connectivity
    src_nodes = []
    dst_nodes = []

    # Add edges only where there's actual connectivity in region_matrix
    local_nodes = [region_id] + list(neighbors)
    for i, src in enumerate(local_nodes):
        for j, dst in enumerate(local_nodes):
            if region_matrix[src, dst] > 0:
                src_nodes.append(i)
                dst_nodes.append(j)

    # Create DGL graph
    g = dgl.graph((torch.tensor(src_nodes), torch.tensor(dst_nodes)))

    num_nodes = len(neighbors) + 1
    feature_dim = 64

    # Initialize spatial features with just the number of adjacent regions
    spatial_features = torch.zeros((num_nodes, feature_dim))
    for i, node in enumerate(local_nodes):
        # Count number of adjacent regions for each node
        num_adjacent = len(np.where(region_matrix[node] > 0)[0])
        spatial_features[i, 0] = num_adjacent  # Put the count in first dimension

    g.ndata['batch_idx'] = torch.zeros(num_nodes, dtype=torch.long)
    g.ndata['spatial_idx'] = torch.tensor(local_nodes, dtype=torch.long)
    g.ndata['spatial_feature'] = spatial_features

    # print(f"Created local graph with {g.number_of_nodes()} nodes and {g.number_of_edges()} edges")
    return g


def compute_loss(pred, true):
    # """
    # Compute MSE
    # Args:
    #     pred: predicted values[batch_size, num_edges, features]
    #     true: true values [batch_size, num_edges, features]
    # """
    # mse_per_edge = torch.mean((pred - true) ** 2, dim=(0, 2))  # Shape: (891,)
    # return mse_per_edge.mean()/1000
    # # return mse_per_edge.mean()

    """
    Compute robust MSE loss with safety checks
    """
    # Input validation
    # assert torch.isfinite(pred).all(), "Pred contains non-finite values"
    # assert torch.isfinite(true).all(), "True contains non-finite values"

    # Check predictions
    if not torch.isfinite(pred).all():
        print("Pred contains non-finite values")
        print("Non-finite indices:", torch.where(~torch.isfinite(pred)))
        print("Non-finite values:", pred[~torch.isfinite(pred)])
        return 0
    # else:
    #     print("All pred values are finite")

    # Check ground truth
    if not torch.isfinite(true).all():
        print("True contains non-finite values")
        print("Non-finite indices:", torch.where(~torch.isfinite(true)))
        print("Non-finite values:", true[~torch.isfinite(true)])
    # else:
    #     print("All true values are finite")

    # Clip extremely large values
    pred = torch.clamp(pred, -1e6, 1e6)
    true = torch.clamp(true, -1e6, 1e6)

    # Compute difference
    diff = pred - true

    # Compute squared difference with safety checks
    squared = torch.clamp(diff ** 2, 0, 1e6)

    # Compute mean with safety checks
    mse_per_edge = torch.mean(squared, dim=(0, 2))
    final_loss = mse_per_edge.mean()

    # Final validation
    assert torch.isfinite(final_loss), "Loss is not finite"

    # return final_loss

    return final_loss/1000


def compute_metrics(pred, true):
    epsilon = 1e-7
    # """
    # Compute MAE, RMSE and MAPE
    # Args:
    #     pred: predicted values[batch_size, num_edges, features]
    #     true: true values [batch_size, num_edges, features]
    # """
    # # ensure inputs are on cpu and converted to numpy
    # pred = pred.detach().cpu().numpy()
    # true = true.detach().cpu().numpy()
    #
    # # Compute metrics across all dimensions
    # mae = np.mean(np.abs(pred - true))
    # rmse = np.sqrt(np.mean(np.square(pred - true)))
    # mape = np.mean(np.abs((pred - true) / (true + 1e-5))) * 100
    #
    # # # If you want metrics per feature (across batch and edges)
    # # mae_per_feature = np.mean(np.abs(pred - true), axis=(0, 1))  # shape: (3,)
    # # rmse_per_feature = np.sqrt(np.mean(np.square(pred - true), axis=(0, 1)))  # shape: (3,)
    # # mape_per_feature = np.mean(np.abs((pred - true) / (true + 1e-5)), axis=(0, 1)) * 100  # shape: (3,)
    #
    # # # If you want metrics per edge (across batch and features)
    # # mae_per_edge = np.mean(np.abs(pred - true), axis=(0, 2))  # shape: (891,)
    # # rmse_per_edge = np.sqrt(np.mean(np.square(pred - true), axis=(0, 2)))  # shape: (891,)
    # # mape_per_edge = np.mean(np.abs((pred - true) / (true + 1e-5)), axis=(0, 2)) * 100  # shape: (891,)
    #
    # # return mae, rmse, mape, mae_per_feature, rmse_per_feature, mape_per_feature
    # return mae/1000, rmse/1000, mape/1000
    # # return mae, rmse, mape

    """
    Compute robust metrics with safety checks
    """
    # Convert to numpy with safety checks
    pred = np.clip(pred.detach().cpu().numpy(), -1e6, 1e6)
    true = np.clip(true.detach().cpu().numpy(), -1e6, 1e6)

    # Compute metrics with safety checks
    mae = np.mean(np.abs(pred - true))
    rmse = np.sqrt(np.mean(np.square(pred - true)) + epsilon)
    mape = np.mean(np.abs((pred - true) / (np.abs(true) + epsilon))) * 100

    # Validate results
    # assert np.isfinite(mae), "MAE is not finite"
    # assert np.isfinite(rmse), "RMSE is not finite"
    # assert np.isfinite(mape), "MAPE is not finite"
    if not np.isfinite(mae) or not np.isfinite(rmse) or not np.isfinite(mape):
        print(f"Non-finite metric detected - MAE: {mae}, RMSE: {rmse}, MAPE: {mape}")
        return 0, 0, 0

    # return mae, rmse, mape
    return mae/1000, rmse/1000, mape/1000


def train(model, train_loader, valid_loader, region_matrix, global_graph, opt, device):
    # Enable gradient scaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler()

    torch.cuda.empty_cache()
    tb_writer = SummaryWriter(log_dir=os.path.join(opt.model_save_path, 'tensorboard'))
    cpu_ram_records = []

    lr = opt.lr_base
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=opt.lr_step,
                                                           factor=opt.lr_decay, threshold=1e-3)
    # criterion = nn.MSELoss()

    trained_epoch = 0
    best_val_loss = float('inf')
    start_time = time.time()

    print("Training...")
    for epoch_i in range(opt.epochs):
        torch.cuda.empty_cache()
        print("=========Epoch: {}=========".format(epoch_i))
        trained_epoch += 1
        model.train()
        train_mae = train_rmse = train_mape = 0
        train_loss = 0
        train_batches = 0

        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)

            # Get batch data
            region_hist = batch['region_hist']  # Dict[region_id -> Tensor[num_edges_in_region, hist_len]]
            # region_matrix = batch['region_matrix']  # [num_regions, num_regions]
            # region_future = model(batch['region_future'])  # Dict[region_id -> Tensor[batch_size, num_edges, pred_len]]
            # print("region_matrix type:", type(region_matrix))
            # print("region_matrix.shape:", region_matrix.shape)

            # break down loss for one batch
            batch_loss = 0
            batch_mae = batch_rmse = batch_mape = 0
            region_count = 0
            # process for each region
            monitor_thread = monitor_memory(interval=0.5)

            # for region_id in list(region_hist.keys())[:50]:
            for region_id in list(region_hist.keys()):
                # get data for current region
                hist_data = region_hist[region_id].to(device)
                # print("hist_data.size(1):", hist_data.size(1))
                if hist_data.size(1) == 0 or hist_data.size(1) >= 2000:
                    continue
                else:
                    print("region_id:", region_id, "edge_size:", hist_data.size(1), "\t")

                    region_count += 1
                    future_data = batch['region_future'][region_id].to(device)
                    normalized_future = normalize_ground_truth(future_data)
                    # create local graph
                    # edge_neighbors = new_edge2edge_neighbor_dicts[region_id]
                    local_graph = create_local_graph(region_id, region_matrix)
                    local_graph = local_graph.to(device)

                    # Get global spatial indices
                    # local_spatial_idx = get_spatial_indices(region_id, region_matrix).to(device)
                    with torch.cuda.amp.autocast():
                        # Forward pass for this region
                        pred = model(global_graph, opt.time_slot_num, local_graph, hist_data)

                        # calculate loss and metrics
                        # region_loss = criterion(pred, normalized_future)
                        avg_region_loss = compute_loss(pred, normalized_future)
                        batch_loss += avg_region_loss / region_count

                    # Compute metrics for this region
                    mae, rmse, mape = compute_metrics(
                        pred.reshape(-1, pred.size(-1)),
                        future_data.reshape(-1, future_data.size(-1))
                    )
                    batch_mae += mae
                    batch_rmse += rmse
                    batch_mape += mape

            print(f'Train Batch - Loss: {batch_loss:.4f}, MAE: {batch_mae:.4f}, '
                          f'RMSE: {batch_rmse:.4f}, MAPE: {batch_mape:.4f}')

            stop_monitoring(monitor_thread)
            print("Memory usage after prediction: {:.2f} MB".format(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024))

            # Proper gradient scaling and clipping
            scaler.scale(batch_loss).backward()

            # Unscale before clip_grad_norm
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.max_grad_norm)
            # Step with gradient scaling
            scaler.step(optimizer)
            scaler.update()

            # Check for gradient issues
            if not check_gradients(model):
                print("Warning: Gradient issues detected")
                continue

            # # Average loss and metrics across regions
            # num_regions = region_count
            # batch_loss /= num_regions
            # batch_mae /= num_regions
            # batch_rmse /= num_regions
            # batch_mape /= num_regions
            #
            # # Backward pass
            # # batch_loss.backward()
            # # torch.nn.utils.clip_grad_norm_(model.parameters(), opt.max_grad_norm)
            # scaler.scale(batch_loss).backward()
            # scaler.step(optimizer)
            # scaler.update()
            # optimizer.step()

            # Accumulate metrics
            train_loss += batch_loss.item()
            train_mae += batch_mae
            train_rmse += batch_rmse
            train_mape += batch_mape
            train_batches += 1
            # Print metrics
        print(f'Epoch Train - Loss: {train_loss:.4f}, MAE: {train_mae:.4f}, '
              f'RMSE: {train_rmse:.4f}, MAPE: {train_mape:.4f}')


        # Validation
        print("Validation...")
        model.eval()
        val_loss = val_mae = val_rmse = val_mape = 0
        val_batches = 0

        with torch.no_grad():
            for batch in valid_loader:
                region_hist = batch['region_hist']
                # region_matrix = batch['region_matrix']
                # region_future = batch['region_future']

                batch_loss = 0
                batch_mae = batch_rmse = batch_mape = 0
                region_count = 0
                # process for each region
                for region_id in region_hist.keys():
                    hist_data = region_hist[region_id].to(device)
                    if hist_data.size(1) == 0 or hist_data.size(1) >= 5000:
                        continue
                    else:
                        print("region_id:", region_id, "edge_size:", hist_data.size(1))

                        region_count += 1
                        future_data = batch['region_future'][region_id].to(device)
                        normalized_future = normalize_ground_truth(future_data)

                        local_graph = create_local_graph(region_id, region_matrix)
                        local_graph = local_graph.to(device)

                        # global_spatial_idx = get_spatial_indices(region_id, region_matrix)
                        # time_slot_size, batch_local_graph, traffic_h, global_spatial_idx)
                        with torch.cuda.amp.autocast():
                            # Forward pass for this region
                            pred = model(global_graph, opt.time_slot_num, local_graph, hist_data)

                            avg_region_loss = compute_loss(pred, normalized_future)
                            batch_loss += avg_region_loss

                        mae, rmse, mape = compute_metrics(
                            pred.reshape(-1, pred.size(-1)),
                            future_data.reshape(-1, future_data.size(-1))
                        )

                        batch_mae += mae
                        batch_rmse += rmse
                        batch_mape += batch_mape
                print(f'Validation Batch - Loss: {batch_loss:.4f}, MAE: {batch_mae:.4f}, '
                          f'RMSE: {batch_rmse:.4f}, MAPE: {batch_mape:.4f}')

                num_regions = region_count
                batch_loss /= num_regions
                batch_mae /= num_regions
                batch_rmse /= num_regions
                batch_mape /= num_regions

                val_loss += batch_loss.item()
                val_mae += batch_mae
                val_rmse += batch_rmse
                val_mape += batch_mape
                val_batches += 1

        # Calculate average metrics
        if train_batches > 0:
            avg_train_loss = train_loss / train_batches
            avg_train_mae = train_mae / train_batches
            avg_train_rmse = train_rmse / train_batches
            avg_train_mape = train_mape / train_batches

        if val_batches > 0:
            avg_val_loss = val_loss / val_batches
            avg_val_mae = val_mae / val_batches
            avg_val_rmse = val_rmse / val_batches
            avg_val_mape = val_mape / val_batches

        # Print metrics
        print(f'Epoch {epoch_i}:')
        print(f'Train - Loss: {avg_train_loss:.4f}, MAE: {avg_train_mae:.4f}, '
              f'RMSE: {avg_train_rmse:.4f}, MAPE: {avg_train_mape:.4f}')
        print(f'Val - Loss: {avg_val_loss:.4f}, MAE: {avg_val_mae:.4f}, '
              f'RMSE: {avg_val_rmse:.4f}, MAPE: {avg_val_mape:.4f}')

        # Save best model based on validation loss

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            metrics = {
                'val_loss': best_val_loss,
                'val_mae': avg_val_mae,
                'val_rmse': avg_val_rmse,
                'val_mape': avg_val_mape
            }
            save_path = os.path.join(opt.model_save_path, 'model_acchigh.ckpt'),
            save_model(model, epoch_i, metrics, save_path)

        #
        #     torch.save({
        #         'epoch': epoch_i,
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'val_loss': best_val_loss,
        #         'val_mae': avg_val_mae,
        #         'val_rmse': avg_val_rmse,
        #         'val_mape': avg_val_mape
        #     }, opt.model_save_path)
        # print('    - [Info] The checkpoint file (Valid Loss Low) has been updated.')

        # Log metrics
        lr = optimizer.param_groups[0]['lr']
        tb_writer.add_scalar('learning_rate', lr, epoch_i)

        # Monitor CPU RAM usage
        cpu_ram = psutil.Process(os.getpid()).memory_info().rss
        cpu_ram_records.append(cpu_ram)
        tb_writer.add_scalar('cpu_ram', round(cpu_ram * 1.0 / 1024 / 1024, 3), epoch_i)

        # Learning rate scheduling
        scheduler.step(avg_val_loss)

        # Early stop
        if lr <= 0.9 * 1e-5:
            print("==> [Info] Early Stop since lr is too small After Epoch {}.".format(epoch_i))
            break

    print("[Info] Training Finished, using {:.3f}s for {} epochs".format(time.time() - start_time, trained_epoch))
    tb_writer.close()

    model_size = sys.getsizeof(model.parameters())
    print("model size: {} Bytes".format(model_size))

    # gpu_ram = torch.cuda.memory_stats(device=device)['active_bytes.all.peak']
    # print("peak gpu memory usage: {:.3f} MB".format(gpu_ram * 1.0 / 1024 / 1024))

    cpu_ram_peak = max(cpu_ram_records)
    print("current memory usage: {:.3f} MB".format(cpu_ram_peak * 1.0 / 1024 / 1024))


def eval_epoch(model, valid_data, global_graph, region_matrix, device, opt):
    model.eval()
    test_loss = test_mae = test_rmse = test_mape = 0
    test_batches = 0

    with torch.no_grad():
        for batch in valid_data:
            region_hist = batch['region_hist']
            # region_matrix = batch['region_matrix']
            # region_future = batch['region_future']

            batch_loss = 0
            batch_mae = batch_rmse = batch_mape = 0
            region_count = 0
            # process for each region
            for region_id in region_hist.keys():
                hist_data = region_hist[region_id].to(device)
                if hist_data.size(1) == 0 or hist_data.size(1) >= 5000:
                    continue
                else:
                    print("region_id:", region_id, "edge_size:", hist_data.size(1))

                    region_count += 1
                    future_data = batch['region_future'][region_id].to(device)
                    normalized_future = normalize_ground_truth(future_data)

                    local_graph = create_local_graph(region_id, region_matrix)
                    local_graph = local_graph.to(device)

                    # global_spatial_idx = get_spatial_indices(region_id, region_matrix)
                    # time_slot_size, batch_local_graph, traffic_h, global_spatial_idx)
                    with torch.cuda.amp.autocast():
                        # Forward pass for this region
                        pred = model(global_graph, opt.time_slot_num, local_graph, hist_data)

                        avg_region_loss = compute_loss(pred, normalized_future)
                        batch_loss += avg_region_loss

                    mae, rmse, mape = compute_metrics(
                        pred.reshape(-1, pred.size(-1)),
                        future_data.reshape(-1, future_data.size(-1))
                    )

                    batch_mae += mae
                    batch_rmse += rmse
                    batch_mape += batch_mape
                print(f'Test Batch - Loss: {batch_loss:.4f}, MAE: {batch_mae:.4f}, '
                      f'RMSE: {batch_rmse:.4f}, MAPE: {batch_mape:.4f}')

            num_regions = region_count
            batch_loss /= num_regions
            batch_mae /= num_regions
            batch_rmse /= num_regions
            batch_mape /= num_regions

            test_loss += batch_loss.item()
            test_mae += batch_mae
            test_rmse += batch_rmse
            test_mape += batch_mape
            test_batches += 1
            print(f'Test Batch - Loss: {batch_loss:.4f}, MAE: {batch_mae:.4f}, '
                  f'RMSE: {batch_rmse:.4f}, MAPE: {batch_mape:.4f}%')
        # Calculate average metrics
    if test_batches > 0:
        avg_test_loss = test_loss / test_batches
        avg_test_mae = test_mae / test_batches
        avg_test_rmse = test_rmse / test_batches
        avg_test_mape = test_mape / test_batches

    return {
        'Loss': avg_test_loss,
        'MAE': avg_test_mae,
        'MAPE': avg_test_rmse,
        'RMSE': avg_test_mape
    }


def model_testing(model, model_path, device, test_data, opt, global_graph, region_matrix):
    model, metrics = load_model(model, model_path, device)

    print("Test Size: {}".format(len(test_data)))
    test_loader = DataLoader(dataset=test_data, batch_size=opt.batch_size,
                             shuffle=False, drop_last=False, num_workers=2, pin_memory=True)

    test_metrics = eval_epoch(model, test_loader, global_graph, region_matrix, device, opt)
    print("\nTest Set Results:")
    print(f"Loss: {test_metrics['Loss']:.4f}")
    print(f"MAE: {test_metrics['MAE']:.4f}")
    print(f"MAPE: {test_metrics['MAPE']:.4f}")
    print(f"RMSE: {test_metrics['RMSE']:.4f}")
    return test_metrics


def save_model(model, epoch, metrics, save_path):
    """
    Simple save function
    metrics: dictionary containing val_loss, val_mae, val_rmse, val_mape
    """
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            # 'optimizer_state_dict': optimizer.state_dict(),
            **metrics  # unpack metrics dict
        }, save_path)
        print(f'    - [Info] Model saved to {save_path}')
        return True
    except Exception as e:
        print(f'    - [Error] Failed to save model: {str(e)}')
        return False


def load_model(model, load_path, device):
    """
    Simple load function
    Returns: model, optimizer, epoch, metrics
    """

    checkpoint = torch.load(load_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # epoch = checkpoint['epoch']
    metrics = {
        'val_loss': checkpoint.get('val_loss', float('inf')),
        'val_mae': checkpoint.get('val_mae', 0),
        'val_rmse': checkpoint.get('val_rmse', 0),
        'val_mape': checkpoint.get('val_mape', 0)
    }
    # print(f'    - [Info] Model loaded from epoch {epoch}')
    return model, metrics


def main():
    parser = argparse.ArgumentParser()
    # dataset parameters
    parser.add_argument('-city', type=str,
                        choices=['Beijing', 'Porto'], default='Beijing')
    parser.add_argument('-gridGra', type=int, default=10)
    parser.add_argument('-region_nums', type=int, default=100)
    parser.add_argument('-edge_nums', type=int, default=651748)
    parser.add_argument('-time_slot_size', type=int, default=288)
    parser.add_argument('-workspace', type=str, default="/data5/edenjingzhao/RTSP/data/Beijing/")
    parser.add_argument('-data_path', type=str, default="/home/edenjingzhao/RegionData/")
    parser.add_argument('-model_save_path', type=str, default='/data5/edenjingzhao/RTSP/data/Beijing/10/model')
    # split parameters
    parser.add_argument('-train_ratio', type=float, default=0.5)
    parser.add_argument('-val_ratio', type=float, default=0.1)
    # running parameters
    parser.add_argument('-seed', type=int, default=42)
    parser.add_argument('-gpu_id', type=str, default="0")
    parser.add_argument("-cpu", action="store_true", dest="force_cpu")
    # training parameters
    parser.add_argument('-batch_size', type=int, default=16)
    parser.add_argument('-epochs', type=int, default=1)
    parser.add_argument('-lr_base', type=float, default=1e-3)
    parser.add_argument('-lr_step', type=int, default=2)
    parser.add_argument('-lr_decay', type=float, default=0.8)
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Maximum gradient norm for gradient clipping')
    # model parameters
    parser.add_argument('-graph_layer', type=int, default=1)
    parser.add_argument('-rnn_layer', type=int, default=1)
    parser.add_argument('-hist_len', type=int, default=20)
    parser.add_argument('-pred_len', type=int, default=3)
    # dimension parameters
    parser.add_argument('-in_dim', type=int, default=64)
    parser.add_argument('-spatial_feature_dim', type=int, default=64, help="dimension of spatial feature")
    parser.add_argument('-temporal_feature_dim', type=int, default=64, help="dimension of temporal feature")
    parser.add_argument('-time_slot_num', type=int, default=288, help="number of global time slots")
    parser.add_argument('-hidden_size', type=int, default=64, help="hidden size")
    # for ablation study
    parser.add_argument('-use_global', type=bool, default=True)
    parser.add_argument('-use_local', type=bool, default=True)
    parser.add_argument('-use_fusion', type=bool, default=True)

    opt = parser.parse_args()
    print(opt)

    # ensure code reproducibility
    if opt.seed is not None:
        torch.manual_seed(opt.seed)
        torch.backends.cudnn.benchmark = False
        np.random.seed(opt.seed)
        random.seed(opt.seed)

    if not opt.model_save_path:
        print('No experiment result will be saved.')
        raise

    if not os.path.exists(opt.model_save_path):
        os.makedirs(opt.model_save_path)

    device = torch.device(
        "cuda:{}".format(opt.gpu_id) if ((not opt.force_cpu) and torch.cuda.is_available()) else "cpu")
    print("running this on {}".format(device))


    # ========= Loading Dataset ========= #
    t0 = time.time()
    speed_data, regions_edges, region_matrix, old2new_edge_dicts, \
        new2old_edge_dicts, new_edge2edge_neighbor_dicts = load_data(speed_path=opt.workspace,
                                                                     region_matrix_path=opt.workspace,
                                                                     region_edges_path=opt.data_path,
                                                                     gridGra=opt.gridGra,
                                                                     time_slot_size=opt.time_slot_size,
                                                                     edge_mapping_path=opt.workspace)
    # create region_edges
    # region_edges = create_region_graph(region_matrix)

    # create global region graph
    global_graph = create_global_graph(region_matrix)
    RTSP_config = RTSPConfig(opt, global_graph)

    print("loading all initial data use {:.3f}s".format(time.time() - t0))

    # Split indices (i.e., get end indices)
    train_end = int(opt.train_ratio * opt.time_slot_size)
    val_end = int((opt.train_ratio + opt.val_ratio) * opt.time_slot_size)
    print("train_end idx: ", train_end, "\t val_end idx: ", val_end)

    # Create datasets
    t0 = time.time()
    train_dataset = TrafficDataset(
        speed_data, regions_edges,
        opt.hist_len, opt.pred_len, 0, train_end)
    print("loading train data use {:.3f}s".format(time.time() - t0))

    t0 = time.time()
    val_dataset = TrafficDataset(
        speed_data, regions_edges,
        opt.hist_len, opt.pred_len, train_end, val_end)
    print("loading validation data use {:.3f}s".format(time.time() - t0))

    t0 = time.time()
    test_dataset = TrafficDataset(
        speed_data, regions_edges,
        opt.hist_len, opt.pred_len, val_end, opt.time_slot_size)

    print("loading test data use {:.3f}s".format(time.time() - t0))

    print("Training Size: {}, Validation Size: {}".format(len(train_dataset), len(val_dataset)))

    # Create dataloaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True,
                              drop_last=False, num_workers=4, pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=opt.batch_size,
                            drop_last=False, num_workers=4, pin_memory=True)
    # test_loader = DataLoader(dataset=test_dataset, batch_size=opt.batch_size)

    # ========= Training Model ========= #
    # Initialize model
    # model = RegionTSP(global_graph=global_graph, time_slot_size=opt.time_slot_size, in_dim=opt.in_dim, spatial_feature_dim=opt.spatial_feature_dim,
    #                   out_spatial_dim=opt.spatial_feature_dim, out_temporal_dim=opt.temporal_feature_dim,
    #                   graph_layers=opt.graph_layer, rnn_layers=opt.rnn_layer, spatial_context_dim=opt.spatial_feature_dim,
    #                   temporal_context_dim=opt.temporal_feature_dim, region_nums=opt.region_nums, edge_nums=opt.edge_nums,
    #                   hidden_size=opt.hidden_size, pred_len=opt.pred_len, device=device)
    # Initialize model
    model = RegionTSP(**RTSP_config.get_config(), device=device)
    model.to(device)

    # train_utils = DataNormalizer(device)
    # train_utils.fit_scaler(train_loader)

    t0 = time.time()
    train(model, train_loader, val_loader, region_matrix, global_graph, opt, device)
    print("[Info] Model Training Finished!")
    print("Training use {:.3f}s".format(time.time() - t0))

    # ========= Testing Model ========= #
    print("[Info] Test Starting...")
    print("=====> AccHigh, Training")
    # t0 = time.time()
    # model_testing(device,
    #               model_path=os.path.join(opt.model_save_path, 'model_acchigh.ckpt'),
    #               test_data=train_dataset,
    #               opt=opt)
    # print("Average training query use {:.3f}s".format((time.time() - t0)/len(train_dataset)))

    t0 = time.time()
    print("=====> AccHigh, Test")
    model_testing(model,
                  model_path=os.path.join(opt.model_save_path, 'model_acchigh.ckpt'),
                  device=device,
                  test_data=test_dataset, opt=opt,
                  global_graph=global_graph, region_matrix=region_matrix)
    print("Average testing use {:.3f}s".format((time.time() - t0)/len(test_dataset)))


if __name__ == '__main__':
    main()