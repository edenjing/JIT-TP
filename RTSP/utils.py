#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：RTSP 
@File    ：utils.py
@Author  ：Eden
@Date    ：2024/11/4 1:39 PM 
'''

import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
import time



class Dict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


def dict_to_object(dictObj):
    if not isinstance(dictObj, dict):
        return dictObj
    inst = Dict()
    for k, v in dictObj.items():
        inst[k] = dict_to_object(v)
    return inst


def normalize_ground_truth(future_data):
    mean = torch.mean(future_data)
    std = torch.std(future_data)

    normalized_future = (future_data - mean) / (std + 1e-8)  # adding small epsilon for numerical stability
    return normalized_future


import psutil
import time
from datetime import datetime
import os
import threading
import torch
import numpy as np


def monitor_memory(interval=1, output_file="memory_log.txt"):
    """
    Monitor system memory usage in real-time
    """

    def memory_logger():
        process = psutil.Process(os.getpid())

        with open(output_file, "w") as f:
            f.write("Timestamp,CPU%,MemoryUsage(MB),SystemMemoryUsage%,GPUMemoryUsage(MB)\n")

            while not stop_flag.is_set():
                try:
                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    cpu_percent = process.cpu_percent()
                    memory_info = process.memory_info()
                    memory_usage_mb = memory_info.rss / 1024 / 1024
                    system_memory = psutil.virtual_memory().percent

                    # Get GPU memory if using PyTorch
                    if torch.cuda.is_available():
                        # gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
                        gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024 / 1024 # GB
                    else:
                        gpu_memory = 0

                    f.write(f"{current_time},{cpu_percent:.1f},{memory_usage_mb:.2f},"
                            f"{system_memory:.1f},{gpu_memory:.2f}\n")
                    print(f"{current_time},{cpu_percent:.1f},{memory_usage_mb:.2f},{system_memory:.1f},{gpu_memory:.2f}")
                    f.flush()

                    time.sleep(interval)

                except Exception as e:
                    print(f"Error in memory monitoring: {e}")
                    break

    global stop_flag
    stop_flag = threading.Event()

    monitor_thread = threading.Thread(target=memory_logger)
    monitor_thread.start()

    return monitor_thread


def stop_monitoring(thread):
    stop_flag.set()
    thread.join()