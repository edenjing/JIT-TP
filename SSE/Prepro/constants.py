#!/usr/bin/env python
# -*- coding: UTF-8 -*-

dataset = "Beijing"
root_path = '/data5/edenjingzhao/QDTP/data/'

# Porto path
porto_raw_path = root_path + 'PortoData/raw/'
porto_path = root_path + 'PortoData/por_data/'

# Beijing path
beijing_raw_path = root_path + 'BJData/raw/'
beijing_path = root_path + 'BJData/bj_data/'
bj_traj_path = '/data5/TrajectoryData/BeijingTrajectoryMatched'
bj_new_generated_path = '/data5/TrajectoryData/BeijingTrajectoryMatched/NewGeneratedFiles/'
BJ_min_long = 115.375
BJ_min_lat = 39.4167
BJ_max_long = 117.5
BJ_max_lat = 41.0833

# Beijing road network (fourth ring road)
# BJ_min_long = 116.291873
# BJ_min_lat = 39.836084
# BJ_max_long = 116.485619
# BJ_max_lat = 39.995464

# speed profile generation
traj_start_day = 7
traj_dir_num = 10
SP_days = 9
time_slot_duration = 15  # min

# query generation
sp_tsp = 0
traj_json_day = 10
min_roadNum = 10

# split and label
train_ratio = 0.6
valid_ratio = 0.2

#--------------------------------------#
bj_sp_min = 0
bj_sp_max = 1128
bj_nodeNum = 296710
INF = 2147483647

# split road network into regions
coarse_gra = 10
fine_gra = 3
# generate queries
node_num_in_fine_grid = 1  # fine_grid * fine_grid

porto_nodenum = 7328
porto_sp_min = 0
porto_sp_max = 7

# region importance parameter
region_pro_para = 30
