#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
Train SSE model
Tasks:
1. region embedding
    - spatial coordinates & sp info
2. weighted loss
    - time density / proximity weights
'''

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import argparse
import os
import pickle
import random
import sys
import time
import psutil
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from ss_grids import FeatureGenerator, QryRegData, SSRegionPred
from conf import ssgrid_para
from utils import dict_to_object
import numpy as np


# metrics: recall, precision, f1, Jaccard
# def cal_rec(pred, onehot_target):
#     recall_values = []
#     precision_values = []
#     F1_values = []
#     for i in range(pred.size(0)):
#         pred_sample = pred[i]
#         target_sample = onehot_target[i]
#
#         target_indices = torch.nonzero(target_sample).reshape(-1)
#         pred_indices = torch.argsort(pred_sample, descending=True)
#         pred_indices = pred_indices[:target_indices.numel()]
#
#         intersection = torch.tensor(list(set(target_indices.tolist()).intersection(set(pred_indices.tolist()))))
#
#         recall_value = intersection.size(0) / len(target_indices) if len(target_indices) > 0 else 0
#         precision_value = intersection.size(0) / len(pred_indices) if len(pred_indices) > 0 else 0
#         f1_value = (2 * recall_value * precision_value) / (recall_value + precision_value) if (recall_value + precision_value) > 0 else 0
#
#         recall_values.append(recall_value)
#         precision_values.append(precision_value)
#         F1_values.append(f1_value)
#
#     avg_rec = sum(recall_values)/len(recall_values)
#     avg_pre = sum(precision_values) / len(precision_values)
#     avg_f1 = sum(F1_values) / len(F1_values)
#
#     # print("avg_rec: {}, avg_pre: {}, avg_f1: {}".format(avg_rec, avg_pre, avg_f1))
#     return avg_rec, avg_pre, avg_f1


def cal_acc(pred, onehot_target, gama):
    recall_values = []
    precision_values = []
    F1_values = []
    jac_values = []

    for i in range(pred.size(0)):
        pred_sample = pred[i]
        target_sample = onehot_target[i]
        new_target_sample = target_sample.tolist()

        new_pred_sample = []
        # each query
        for prob in pred_sample:
            if prob > gama:
                new_pred_sample.append(1)
            else:
                new_pred_sample.append(0)

        TP = sum((y_t == 1) and (y_p == 1) for y_t, y_p in zip(new_target_sample, new_pred_sample))
        FP = sum((y_t == 0) and (y_p == 1) for y_t, y_p in zip(new_target_sample, new_pred_sample))
        FN = sum((y_t == 1) and (y_p == 0) for y_t, y_p in zip(new_target_sample, new_pred_sample))
        TN = sum((y_t == 0) and (y_p == 0) for y_t, y_p in zip(new_target_sample, new_pred_sample))

        # calculate metrics
        epsilon = 1e-7
        precision = TP / (TP + FP + epsilon)
        recall = TP / (TP + FN + epsilon)
        f1 = 2 * (precision * recall) / (precision + recall + epsilon)
        jaccard = TP / (TP + FP + FN + epsilon)

        precision_values.append(precision)
        recall_values.append(recall)
        F1_values.append(f1)
        jac_values.append(jaccard)

    avg_pre = sum(precision_values) / len(precision_values)
    avg_rec = sum(recall_values)/len(recall_values)
    avg_f1 = sum(F1_values) / len(F1_values)
    avg_jac = sum(jac_values) / len(jac_values)

    return avg_pre, avg_rec, avg_f1, avg_jac

# with weight
# def cal_loss_BCE(outputs, onehot_target, weight):
#     loss_fn = nn.BCELoss(weight=weight)
#     onehot_target = onehot_target.to(torch.float32)
#     bce_loss = loss_fn(outputs, onehot_target)
#     return bce_loss


# without weight
def cal_loss_BCE(outputs, onehot_target):
    loss_fn = nn.BCELoss()
    onehot_target = onehot_target.to(torch.float32)
    bce_loss = loss_fn(outputs, onehot_target)
    return bce_loss


# data: odt, od_cords, node_seq, node_cords_seq, reg_seq, reg_cords_seq, edge_seq, label, regions
def epoch_forward(data, model, device, gama):
    # o, d, t, o_reg, d_reg, label, regions = data
    o, d, o_reg, d_reg, label, regions, weights = data

    o = o.to(device, non_blocking=True)
    d = d.to(device, non_blocking=True)
    # t = t.to(device, non_blocking=True)
    o_reg = o_reg.to(device, non_blocking=True)
    d_reg = d_reg.to(device, non_blocking=True)
    regions = regions.to(device, non_blocking=True)
    label = label.to(device, non_blocking=True)  # ground-truth label
    weights = weights.to(device, non_blocking=True)

    # outputs = model(o, d, t, o_reg, d_reg, regions, train_phase=True)
    outputs = model(o, d, o_reg, d_reg, regions, weights, train_phase=True)

    # loss = cal_loss_BCE(outputs, label, weights)

    loss = cal_loss_BCE(outputs, label)

    avg_pre, avg_rec, avg_f1, avg_jac = cal_acc(outputs, label, gama)
    return loss, avg_pre, avg_rec, avg_f1, avg_jac


def eval_epoch(model, valid_data, device, gama):
    model.eval()
    # total = 0
    # right = 0
    total_loss = 0
    total_pre = 0
    total_rec = 0
    total_f1 = 0
    total_jac = 0

    batch_num = 0
    with torch.no_grad():
        for data in valid_data:
            loss, avg_pre, avg_rec, avg_f1, avg_jac = epoch_forward(data, model, device, gama)

            total_loss += loss.item()
            total_pre += avg_pre
            total_rec += avg_rec
            total_f1 += avg_f1
            total_jac += avg_jac
            # right += n_correct
            # total += n_word
            batch_num += 1

    # acc1 = right * 1.0 / total
    loss_per_batch = round(total_loss / batch_num, 4)
    pre_per_batch = round(total_pre / batch_num, 4)
    rec_per_batch = round(total_rec / batch_num, 4)
    f1_per_batch = round(total_f1 / batch_num, 4)
    jac_per_batch = round(total_jac / batch_num, 4)

    # print("loss: {}, recall: {}, precision: {}, f1: {}".format(loss_per_batch, rec_per_batch, pre_per_batch, f1_per_batch))
    return loss_per_batch, pre_per_batch, rec_per_batch, f1_per_batch, jac_per_batch


def train(model, train_loader, valid_loader, device, opt, hparams):
    tb_writer = SummaryWriter(log_dir=os.path.join(opt.output_dir, 'tensorboard'))

    cpu_ram_records = []

    lr = opt.lr_base
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=hparams.lr_step,
                                                           factor=hparams.lr_decay, threshold=1e-3)

    best_epoch = 0
    best_epoch_rec = 0
    best_epoch_train = 0
    best_epoch_rec_train = 0
    total_train_step = 0

    train_recs = []
    train_pres = []
    train_f1s = []
    train_jacs = []

    train_losses = []

    valid_recs = []
    valid_pres = []
    valid_f1s = []
    valid_jacs = []

    valid_losses = []
    trained_epoch = 0

    start_time = time.time()
    for epoch_i in range(opt.epochs):
        print("=========Epoch: {}=========".format(epoch_i))
        trained_epoch += 1
        model.train()
        total_train_loss = 0
        total_rec = 0
        total_pre = 0
        total_f1 = 0
        total_jac = 0
        # right = 0
        # total = 0

        batch_num = 0
        for data in train_loader:
            loss, avg_pre, avg_rec, avg_f1, avg_jac = epoch_forward(data, model, device, opt.gama)

            total_train_loss += loss.item()
            # right += n_correct
            # total += n_word
            total_rec += avg_rec
            total_pre += avg_pre
            total_f1 += avg_f1
            total_jac += avg_jac

            batch_num += 1

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_train_step += 1
            # write loss to tensorboard
            tb_writer.add_scalars("Train_loss", {'loss': loss.item()}, total_train_step)

        if epoch_i == 0:
            print('[Info] Single epoch time cost:{}'.format(time.time() - start_time))
        train_loss = round(total_train_loss / batch_num, 4)
        # train_accu = round(right * 1.0 / total, 4)
        train_rec_acc = round(total_rec / batch_num, 4)
        train_pre_acc = round(total_pre / batch_num, 4)
        train_f1_acc = round(total_f1 / batch_num, 4)
        train_jac_acc = round(total_jac / batch_num, 4)

        train_losses += [train_loss]  # add train_loss in different epoches
        train_recs += [train_rec_acc]
        train_pres += [train_pre_acc]
        train_f1s += [train_f1_acc]
        train_jacs += [train_jac_acc]

        print("==> Evaluation")
        valid_loss, valid_pre, valid_rec, valid_f1, valid_jac = eval_epoch(model, valid_loader, device, opt.gama)

        # valid_losses += [round(valid_loss, 4)]
        # valid_coi_accus += [round(valid_coi_acc, 4)]

        valid_losses += [valid_loss]
        valid_recs += [valid_rec]
        valid_pres += [valid_pre]
        valid_f1s += [valid_f1]
        valid_jacs += [valid_jac]

        checkpoint = {'epoch': epoch_i, 'settings': opt, 'params': dict(hparams), 'model': model.state_dict()}

        if train_loss <= min(train_losses):
            best_epoch_train = epoch_i
            torch.save(checkpoint, os.path.join(opt.output_dir, 'model_train.ckpt'))
            print('    - [Info] The checkpoint file (Train Loss Low) has been updated.')
        # if train_coi_acc >= max(train_coi_accus):
        if train_rec_acc >= max(train_recs):
            best_epoch_rec_train = epoch_i
            torch.save(checkpoint, os.path.join(opt.output_dir, 'model_train_acchigh.ckpt'))
            print('    - [Info] The checkpoint file (Train Acc High) has been updated.')
        if round(valid_loss, 4) <= min(valid_losses):
            best_epoch = epoch_i
            torch.save(checkpoint, os.path.join(opt.output_dir, 'model.ckpt'))
            print('    - [Info] The checkpoint file (Loss Low) has been updated.')
        if round(valid_rec, 4) >= max(valid_recs):
            best_epoch_rec = epoch_i
            torch.save(checkpoint, os.path.join(opt.output_dir, 'model_acchigh.ckpt'))
            print('    - [Info] The checkpoint file (Acc High) has been updated.')

        tb_writer.add_scalars('Loss', {'train': total_train_loss / batch_num, 'val': valid_loss}, epoch_i)
        tb_writer.add_scalars('Rec', {'train': total_rec / batch_num, 'val': valid_rec}, epoch_i)
        tb_writer.add_scalars('Pre', {'train': total_pre / batch_num, 'val': valid_pre}, epoch_i)
        tb_writer.add_scalars('F1', {'train': total_f1 / batch_num, 'val': valid_f1}, epoch_i)
        tb_writer.add_scalars('Jac', {'train': total_jac / batch_num, 'val': valid_jac}, epoch_i)

        tb_writer.add_scalar('learning_rate', lr, epoch_i)

        cpu_ram = psutil.Process(os.getpid()).memory_info().rss

        # gpu_ram = torch.cuda.memory_stats(device=device)['active_bytes.all.current']

        cpu_ram_records.append(cpu_ram)

        tb_writer.add_scalar('cpu_ram', round(cpu_ram * 1.0 / 1024 / 1024, 3), epoch_i)
        # tb_writer.add_scalar('gpu_ram', round(gpu_ram * 1.0 / 1024 / 1024, 3), epoch_i)

        # scheduler.step(valid_coi_acc)
        scheduler.step(valid_rec)
        lr_last = lr
        lr = optimizer.param_groups[0]['lr']

        if lr <= 0.9 * 1e-5:
            print("==> [Info] Early Stop since lr is too small After Epoch {}.".format(epoch_i))
            break

    print("[Info] Training Finished, using {:.3f}s for {} epochs".format(time.time() - start_time, trained_epoch))
    tb_writer.close()
    print("[Info] Train Loss lowest epoch: {}, loss: {}, pre: {}, rec: {}, f1: {}, jac: {}".format(best_epoch_train, train_losses[best_epoch_train],
                                                                                                    train_pres[best_epoch_train], train_recs[best_epoch_train],
                                                                                                    train_f1s[best_epoch_train], train_jacs[best_epoch_train]))

    print("[Info] Train rec_acc highest epoch: {}, loss: {}, pre: {}, rec: {}, f1: {}, jac: {}".format(best_epoch_rec_train, train_losses[best_epoch_rec_train],
                                                                                            train_pres[best_epoch_train], train_recs[best_epoch_rec_train],
                                                                                            train_f1s[best_epoch_train], train_jacs[best_epoch_train]))

    print("[Info] Validation Loss lowest epoch: {}, loss: {}, pre: {}, rec: {}, f1: {}, jac: {}".format(best_epoch, valid_losses[best_epoch],
                                                                                                        valid_pres[best_epoch_train], valid_recs[best_epoch_train],
                                                                                                        valid_f1s[best_epoch_train], valid_jacs[best_epoch_train]))

    print("[Info] Validation rec_acc highest epoch: {}, loss: {}, pre: {}, rec: {}, f1: {}, jac: {}".format(best_epoch_rec, valid_losses[best_epoch_rec],
                                                                                            valid_pres[best_epoch_train], valid_recs[best_epoch_rec_train],
                                                                                            valid_f1s[best_epoch_train], valid_jacs[best_epoch_train]))


    model_size = sys.getsizeof(model.parameters())
    print("model size: {} Bytes".format(model_size))

    # gpu_ram = torch.cuda.memory_stats(device=device)['active_bytes.all.peak']
    # print("peak gpu memory usage: {:.3f} MB".format(gpu_ram * 1.0 / 1024 / 1024))

    cpu_ram_peak = max(cpu_ram_records)
    print("current memory usage: {:.3f} MB".format(cpu_ram_peak * 1.0 / 1024 / 1024))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-workspace", type=str, default="/data5/edenjingzhao/QDTP/data/BJData/bj_data/")
    parser.add_argument('--output_dir', type=str, default="/data5/edenjingzhao/QDTP/data/BJData/bj_model/")
    parser.add_argument('-seed', type=int, default=42)
    parser.add_argument('-epochs', type=int, default=300)
    parser.add_argument('-lr_base', type=float, default=1e-3)
    parser.add_argument('-gpu_id', type=str, default="0")
    parser.add_argument('-city', type=str,
                        choices=['beijing', 'Porto'], default='beijing')
    parser.add_argument("-cpu", action="store_true", dest="force_cpu")
    parser.add_argument("-gama", type=float, default=0.5)

    opt = parser.parse_args()
    print(opt)

    # ensure code reproducibility
    if opt.seed is not None:
        torch.manual_seed(opt.seed)
        torch.backends.cudnn.benchmark = False
        np.random.seed(opt.seed)
        random.seed(opt.seed)

    if not opt.output_dir:
        print('No experiment result will be saved.')
        raise

    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)

    device = torch.device(
        "cuda:{}".format(opt.gpu_id) if ((not opt.force_cpu) and torch.cuda.is_available()) else "cpu")
    print("running this on {}".format(device))

    hparams = dict_to_object(ssgrid_para[opt.city])
    hparams.pretrained_node_emb = os.path.join(opt.workspace, hparams.pretrained_node_emb)
    hparams.pretrained_region_emb = os.path.join(opt.workspace, hparams.pretrained_region_emb)
    hparams.device = device
    print(hparams)

    # ========= Loading Dataset ========= #
    processor = FeatureGenerator(opt.workspace,
                                 region_num=hparams.region_num)
    # region_data = processor.get_region_info(opt.workspace)  # all region data

    # train
    t0 = time.time()
    train_data = processor.load4sse("train")
    train_weight = processor.get_weight_info("train")
    train_data = QryRegData(train_data, processor.region_size, train_weight)
    # train_data = QryRegData(train_data, processor.region_size)
    print("loading training data use {:.3f}s".format(time.time() - t0))

    # valid
    t0 = time.time()
    valid_data = processor.load4sse("val")
    valid_weight = processor.get_weight_info("val")
    valid_data = QryRegData(valid_data, processor.region_size, valid_weight)
    # valid_data = QryRegData(valid_data, processor.region_size)
    print("loading validation data use {:.3f}s".format(time.time() - t0))

    print("Training Size: {}, Validation Size: {}".format(len(train_data), len(valid_data)))

    # train_data = DataOperation(train_data)
    # valid_data = DataOperation(valid_data)

    train_loader = DataLoader(dataset=train_data, batch_size=hparams.batch_size,
                              shuffle=True, drop_last=False, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(dataset=valid_data, batch_size=hparams.batch_size,
                              shuffle=False, drop_last=False, num_workers=4, pin_memory=True)

    model = SSRegionPred(hparams).to(device)
    gama = opt.gama

    t0 = time.time()
    train(model, train_loader, valid_loader, device, opt, hparams)
    print("[Info] Model Training Finished!")
    print("Training use {:.3f}s".format(time.time() - t0))

    t0 = time.time()
    test_data = processor.load4sse("test")
    test_weight = processor.get_weight_info("test")
    test_data = QryRegData(test_data, processor.region_size, test_weight)
    # test_data = QryRegData(test_data, processor.region_size)
    print("loading test data use {:.3f}s".format(time.time() - t0))

    print("[Info] Test Starting...")
    print("=====> AccHigh, Training")
    t0 = time.time()
    model_testing(device,
                  model_path=os.path.join(opt.output_dir, 'model_acchigh.ckpt'),
                  test_data=train_data,
                  gama=gama)
    print("Average training query use {:.3f}s".format((time.time() - t0)/len(train_data)))


    t0 = time.time()
    print("=====> AccHigh, Test")
    model_testing(device,
                  model_path=os.path.join(opt.output_dir, 'model_acchigh.ckpt'),
                  test_data=test_data,
                  gama=gama)
    print("Average testing query use {:.3f}s".format((time.time() - t0)/len(test_data)))


def load_model(model_file, device):
    checkpoint = torch.load(model_file, map_location=device)
    hparams = dict_to_object(checkpoint['params'])
    hparams.device = device

    model = SSRegionPred(hparams).to(device)

    model.load_state_dict(checkpoint['model'])
    print('[Info] Trained model state loaded.')
    return model, hparams


# output the estimation of regions
def model_testing(device, model_path, test_data, gama):
    model, hparams = load_model(model_path, device)
    print(hparams)

    print("Test Size: {}".format(len(test_data)))
    test_loader = DataLoader(dataset=test_data, batch_size=hparams.batch_size,
                             shuffle=False, drop_last=False, num_workers=2, pin_memory=True)

    loss, pre, rec, f1, jac = eval_epoch(model, test_loader, device, gama)
    print("loss: {}, pre: {}, rec: {}, f1: {}, jac: {}".format(loss, pre, rec, f1, jac))


if __name__ == '__main__':
    main()