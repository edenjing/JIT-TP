#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
Task:
1. generate speed profile by day
2. patch zero value in the file
'''

from constants import *
import os
import time
import json
from datetime import datetime, timedelta
from collections import defaultdict

def read_trajs(traj_path, traj_start_day):
    all_trajs = {}
    # Iterate over the level 1 directories (1 to 30)
    for i in range(traj_start_day, traj_dir_num + int(traj_start_day)):
        dir_1 = os.path.join(traj_path, str(i))
        trajs = []
        # Check if the directory exists
        if os.path.exists(dir_1) and os.path.isdir(dir_1):
            # Iterate over the files in the current level 1 directory
            for filename in os.listdir(dir_1):
                file_path = os.path.join(dir_1, filename)

                # Check if the file is a txt file
                if os.path.isfile(file_path) and filename.endswith('.txt'):
                    with open(file_path, 'r') as file:
                        traj = []
                        # Read the lines in the file
                        lines = file.readlines()

                        # Process each line and create traces
                        for line in lines:
                            line_parts = line.strip().split(' ')
                            road_id = line_parts[0]
                            intime = line_parts[1]
                            outtime = line_parts[2]
                            road_info = (road_id, intime, outtime)
                            traj.append(road_info)
                        trajs.append(traj)
        all_trajs[i] = trajs
    print("Read trajectories finished.")
    return all_trajs


def read_roadMap(roadMap_path):
    road_id_mapping = {}
    new_road_ids = []

    with open(roadMap_path + "/beijingRoadMap", 'r') as file:
        lines = file.readlines()

        for line in lines:
            line_info = line.strip().split('\t')
            old_road_id = line_info[0]
            new_road_id = line_info[1]
            road_id_mapping[int(old_road_id)] = int(new_road_id)
            new_road_ids.append(int(new_road_id))

    return road_id_mapping, new_road_ids


def read_roadInfo(roadInfo_path):
    road_to_speed = {}

    with open(roadInfo_path + "/beijingRoadInfo", 'r') as file:
        lines = file.readlines()

        for line in lines:
            line_info = line.strip().split('\t')
            road_id = line_info[0]
            speed_limit = line_info[3]
            road_to_speed[int(road_id)] = int(speed_limit)

    return road_to_speed


def read_roadNew(roadNew_path, new_road_ids, road_to_speed, reversed_road_id_mapping):
    max_road_value = max(new_road_ids)
    print("Initial max_road_value: ", max_road_value)

    # store all new road id info
    roads_dict = {}  # road id, direction, length, ID1, ID2
    # store two-way road mapping
    two_way_roads_map = {}

    with open(roadNew_path + "/beijingRoadNew", 'r') as f1:
        first_line = f1.readline().strip(" ")
        edge_num = first_line.split("\n")
        print("Number of edges in old set of ids: ", edge_num[0])

        for line in f1:
            road_info = line.strip(" ")
            if road_info:
                values = road_info.split()
                road_id = values[0]
                if road_id not in roads_dict:
                    roads_dict[int(road_id)] = {}
                roads_dict[int(road_id)]["dir"] = int(values[1])
                roads_dict[int(road_id)]["len"] = int(values[2])
                roads_dict[int(road_id)]["ID1"] = int(values[3])
                roads_dict[int(road_id)]["ID2"] = int(values[4])

                speed_limit = road_to_speed[reversed_road_id_mapping[int(road_id)]]
                roads_dict[int(road_id)]["sl"] = int(speed_limit)

                # increment from the maximum value of the new_road_ids in order
                if int(values[1]) == 0 or int(values[1]) == 1:
                    two_way_road_id = max_road_value + 1
                    two_way_roads_map[int(road_id)] = two_way_road_id

                    if two_way_road_id not in roads_dict:
                        roads_dict[int(two_way_road_id)] = {}
                    roads_dict[int(two_way_road_id)]["dir"] = int(values[1])
                    roads_dict[int(two_way_road_id)]["len"] = int(values[2])
                    roads_dict[int(two_way_road_id)]["ID1"] = int(values[4])
                    roads_dict[int(two_way_road_id)]["ID2"] = int(values[3])
                    roads_dict[int(two_way_road_id)]["sl"] = int(speed_limit)

                    max_road_value += 1

    final_max_road_id = max(roads_dict.keys())
    print("Max road id after mapping: ", final_max_road_id)
    return roads_dict, two_way_roads_map


def read_map(new_path, raw_path):
    roads_path = new_path + "/new_beijingRoadNew.txt"
    coords_path = raw_path + "/beijingNodeNew"

    roads_dict = {}  # road_id, ID1, ID2, length
    nodes_dict = {}  # node_id, longitude, latitude

    with open(roads_path, 'r') as f2:
        first_line = f2.readline().strip(" ")
        edge_num = first_line.split(" ")

        for line in f2:
            road_info = line.strip(" ")
            if road_info:
                road_id, _, length, ID1, ID2, speed_limit = road_info.split(' ')
                if road_id not in roads_dict:
                    roads_dict[int(road_id)] = {}
                roads_dict[int(road_id)]["ID1"] = int(ID1)
                roads_dict[int(road_id)]["ID2"] = int(ID2)
                roads_dict[int(road_id)]["len"] = int(length)
                roads_dict[int(road_id)]["sl"] = int(speed_limit)

    # with open(coords_path, 'r') as f3:
    #     first_line = f3.readline().strip(" ")
    #     _, min_lat, max_lat, min_long, max_long = first_line.split("\t")
    #
    #     for line in f3:
    #         node_info = line.strip("\t")
    #         if node_info:
    #             values = node_info.split()
    #             node_id = values[0]
    #             if node_id not in nodes_dict:
    #                 nodes_dict[int(node_id)] = {}
    #             nodes_dict[int(node_id)]["long"] = float(values[3])
    #             nodes_dict[int(node_id)]["lat"] = float(values[2])
    #
    # return roads_dict, nodes_dict
    return roads_dict


def convert_road_ids(all_trajs, road_id_mapping, two_way_roads_map):
    converted_all_trajs = {}

    for key, trajs in all_trajs.items():
        converted_trajs = []
        for traj in trajs:
            new_traj = []
            for road_info in traj:
                road_id = int(road_info[0])
                intime = int(road_info[1])
                outtime = int(road_info[2])

                if road_id in road_id_mapping:
                    new_road_id = road_id_mapping[road_id]
                    new_road_info = (new_road_id, intime, outtime)
                    new_traj.append(new_road_info)
                elif road_id < 0:
                    new_road_id = -road_id
                    # if negative, find the forward road id, then map to backward road id
                    if new_road_id in road_id_mapping:
                        fwd_road_id = road_id_mapping[new_road_id]
                        bwd_road_id = two_way_roads_map[fwd_road_id]

                        new_road_info = (bwd_road_id, intime, outtime)
                        new_traj.append(new_road_info)
                    else:
                        print("new_road_id not in two_way_roads_map:", new_road_id)
                else:
                    print("Neither road_id in road_id_mapping nor negative value: ", road_id)
            converted_trajs.append(new_traj)
        converted_all_trajs[key] = converted_trajs

    return converted_all_trajs


def write_trajs(all_trajs, store_path):
    for key, value in all_trajs.items():
        trajs = json.dumps(value)
        filename = "Day_" + str(key) + "_trajs.json"
        write_path = os.path.join(store_path, filename)
        if os.path.exists(write_path):
            continue
        else:
            with open(write_path, 'w') as file:
                file.write(trajs)
            print("Saved {}.".format(filename))


def read_json_trajs(new_path, day):
    filename = "Day_" + str(day) + "_trajs.json"
    read_path = os.path.join(new_path, filename)

    if os.path.exists(read_path):
        with open(read_path, 'r') as file:
            trajs = json.load(file)
            return trajs
    else:
        print("File {} does not exist.".format(filename))


def cal_avg_travel_time(Trajs, time_slot_duration, roads_dict):
    # Step 1: Determine the total time domain for all trajectories
    # min_in_time = min(min(record[1] for record in traj) for traj in Trajs)
    # max_out_time = max(max(record[2] for record in traj) for traj in Trajs)

    min_in_time = float('inf')
    max_out_time = float('-inf')
    for trajectory in Trajs:
        for record in trajectory:
            in_time = int(record[1])
            out_time = int(record[2])
            min_in_time = min(min_in_time, in_time)
            max_out_time = max(max_out_time, out_time)
    print("Get min_in_time and max_out_time: ", min_in_time, max_out_time)

    # Step 2: Create time slots with the specified duration within the total time domain
    time_slots = [(current_time, current_time + time_slot_duration * 60) for current_time in range(int(min_in_time), int(max_out_time) + 1, time_slot_duration * 60)]

    print("len(time_slots): ", len(time_slots))

    # Step 3: Calculate total travel time and count for each road segment and time slot
    avg_travel_time = {int(road_id): [[] for _ in range(len(time_slots))] for road_id in roads_dict}
    # avg_travel_time = {}

    # Iterate over trajectories
    for trajectory in Trajs:
        for record in trajectory:
            road_id = int(record[0])
            in_time = int(record[1])
            out_time = int(record[2])
            travel_time = out_time - in_time

            # Find the time slots that include the travel time
            for i, (slot_start, slot_end) in enumerate(time_slots):
                # if slot_start <= in_time <= slot_end or slot_start <= out_time <= slot_end:
                #     avg_travel_time[road_id][i].append(travel_time)
                try:
                    if slot_start <= in_time <= slot_end or slot_start <= out_time <= slot_end:
                        avg_travel_time[road_id][i].append(travel_time)
                        # avg_travel_time.setdefault(road_id, {}).setdefault(i, []).append(travel_time)
                except KeyError:
                    print(f"Error: KeyError occurred for road_id {road_id}")

    # Calculate the average travel time for each road and time slot
    for road_id, travel_time_all_slots in avg_travel_time.items():
        for i in range(len(travel_time_all_slots)):
            if len(travel_time_all_slots[i]) > 0:
                avg_travel_time[road_id][i] = int(sum(travel_time_all_slots[i]) / len(travel_time_all_slots[i]))
            else:
                avg_travel_time[road_id][i] = 0
    print("Compute average travel time for each time slot finished.")
    return avg_travel_time, time_slots


def patch_zero(avg_travel_time, time_slots, roads_dict):
    for edge_id, time_for_slots in avg_travel_time.items():
        for i, time in enumerate(time_for_slots):
            if time == 0:
                # Method 3
                try:
                    speed_limit = sl_mapping(roads_dict[edge_id]['sl']) * 0.277777778  # unit: m/s
                except KeyError:
                    print(f"Error: KeyError occurred for edge_id {edge_id}")

                edge_length = roads_dict[edge_id]['len']
                new_time = edge_length / speed_limit

                if new_time == 0:
                    # Methods 2
                    adj_roads, adj_lens = find_adjacent_edges(edge_id, roads_dict)
                    adj_speeds = [avg_travel_time[road][i]/length for road, length in zip(adj_roads, adj_lens)]
                    # adj_speeds = []
                    # for road, length in zip(adj_roads, adj_lens):
                    #     time = avg_travel_time[road][i]
                    #     speed = time / length    # should be m/s
                    #     adj_speeds.append(speed)

                    avg_speed = sum(adj_speeds) / len(adj_speeds) if adj_speeds else 0
                    new_time = edge_length / avg_speed

                    if new_time == 0:
                        if i - 1 >= 0 and i + 1 < len(time_slots):
                            left_time = avg_travel_time[edge_id][i - 1]
                            right_time = avg_travel_time[edge_id][i + 1]
                            new_time = (left_time + right_time) / 2
                        elif i - 1 < 0:
                            right_time = avg_travel_time[edge_id][i + 1]
                            new_time = right_time
                        else:
                            left_time = avg_travel_time[edge_id][i - 1]
                            new_time = left_time
                avg_travel_time[edge_id][i] = new_time

    return avg_travel_time


# def patch_zero(avg_travel_time, time_slots, roads_dict):
#     for edge_id, time_for_slots in avg_travel_time.items():
#         for i, time in enumerate(time_for_slots):
#             if time == 0:
#
#
#                 new_time = patch_method_three(edge_id, roads_dict) or \
#                            patch_method_two(edge_id, i, avg_travel_time, roads_dict) or\
#                            patch_method_one(edge_id, i, time_slots, avg_travel_time)
#                 avg_travel_time[edge_id][i] = new_time
#     return avg_travel_time


# Method one: 用相邻time slots内的值取平均
def patch_method_one(edge_id, i, time_slots, avg_travel_time):
    if i-1 >= 0 and i+1 < len(time_slots):
        # left_tsl = time_slots[i-1]
        # right_tsl = time_slots[i+1]

        left_time = avg_travel_time[edge_id][i-1]
        right_time = avg_travel_time[edge_id][i+1]
        new_time = (left_time + right_time) / 2
    elif i-1 < 0:
        # right_tsl = time_slots[i+1]
        right_time = avg_travel_time[edge_id][i+1]
        new_time = right_time
    else:
        # left_tsl = time_slots[i-1]
        left_time = avg_travel_time[edge_id][i-1]
        new_time = left_time
    return int(new_time)


# Method two: 同一time slot内，周围邻居路段的速度平均值，再根据距离转成时间
def patch_method_two(edge_id, i, avg_travel_time, roads_dict):
    adj_roads, adj_lens = find_adjacent_edges(edge_id, roads_dict)
    adj_speeds = []
    for road, length in zip(adj_roads, adj_lens):
        time = avg_travel_time[road][i]
        speed = time / length  # should be m/s
        adj_speeds.append(speed)

    avg_speed = sum(adj_speeds) / len(adj_speeds)
    new_time = roads_dict[edge_id]['length'] / avg_speed
    return int(new_time)


# find adjacent edges by road info
def find_adjacent_edges(edge_id, roads_dict):
    ID1 = roads_dict[edge_id]['ID1']
    ID2 = roads_dict[edge_id]['ID2']

    adj_roads = []
    adj_lens = []
    for edge_id in roads_dict:
        if roads_dict[edge_id]['ID2'] == ID1 or roads_dict[edge_id]['ID1'] == ID2:
            adj_roads.append(edge_id)
            adj_lens.append(roads_dict[edge_id]['len'])
    return adj_roads, adj_lens


# 1: 130; 2: 130; 3: 100; 4: 90; 5: 70; 6: 50; 7: 30; 8: 11 (unit: km/h)
def sl_mapping(sl_num):
    if sl_num in {1, 2}:
        return 130
    elif sl_num == 3:
        return 100
    elif sl_num == 4:
        return 90
    elif sl_num == 5:
        return 70
    elif sl_num == 6:
        return 50
    elif sl_num == 7:
        return 30
    else:
        return 11


# method three: 用距离和限速计算
def patch_method_three(edge_id, roads_dict):
    speed = sl_mapping(roads_dict[edge_id]['sl']) * 0.277777778  # unit: m/s

    new_time = roads_dict[edge_id]['len'] / speed
    # if new_time == 0:
        # print("travel time equals to 0 even using speed limit.")
    return int(new_time)


def write_final_sp(average_travel_time, store_path, day):
    write_path = os.path.join(store_path, "BJSP_by_Day", str(day))
    os.makedirs(write_path, exist_ok=True)

    sp_path = os.path.join(write_path, "SP.txt")
    # tsl_path = os.path.join(write_path, "Timeslots.txt")

    # sp
    with open(sp_path, 'w') as f:
        for edge_id, travel_times in average_travel_time.items():
            number_of_travel_times = len(travel_times)
            f.write(f"{edge_id} {number_of_travel_times} ")

            for i, avg_time in enumerate(travel_times):
                avg_time = max(avg_time, 0)
                f.write(f"{i} {avg_time} ")
            f.write("\n")
    print("Saved {}.".format(sp_path))


def write_initial_sp_and_timeslots(average_travel_time, time_slots, store_path, day):
    initial_write_path = os.path.join(store_path, "BJSP_unpatched", str(day))
    os.makedirs(initial_write_path, exist_ok=True)

    sp_path = os.path.join(initial_write_path, "SP.txt")

    # sp
    with open(sp_path, 'w') as f:
        for edge_id, travel_times in average_travel_time.items():
            number_of_travel_times = len(travel_times)
            f.write(f"{edge_id} {number_of_travel_times} ")
            for i, avg_time in enumerate(travel_times):
                avg_time = max(avg_time, 0)
                f.write(f"{i} {avg_time} ")
            f.write("\n")
    print("Saved {}.".format(sp_path))

    tsl_write_path = os.path.join(store_path, "BJSP_by_Day", str(day))
    os.makedirs(tsl_write_path, exist_ok=True)
    tsl_path = os.path.join(tsl_write_path, "Timeslots.txt")

    # timeslots
    with open(tsl_path, "w") as f:
        for slot in time_slots:
            start_tsp = datetime.fromtimestamp(slot[0])
            start_tsp = start_tsp.strftime("%Y-%m-%d-%H-%M-%S")

            end_tsp = datetime.fromtimestamp(slot[1])
            end_tsp = end_tsp.strftime("%Y-%m-%d-%H-%M-%S")

            f.write(f"{start_tsp} {end_tsp}\n")
    print("Saved {}.".format(tsl_path))


def read_initial_sp_and_timeslots(store_path, day):
    sp_path = os.path.join(store_path, "BJSP_unpatched", str(day), "SP.txt")
    tsl_path = os.path.join(store_path, "BJSP_by_Day", str(day), "Timeslots.txt")
    average_travel_time = {}

    with open(sp_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            edge_id, num_times, *time_data = line.split()
            num_times = int(num_times)
            travel_times = [0] * num_times

            for i in range(0, num_times * 2, 2):
                time_slot = int(time_data[i])
                avg_time = int(time_data[i + 1])
                travel_times[time_slot] = avg_time

            average_travel_time[int(edge_id)] = travel_times

    time_slots = []
    with open(tsl_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            start_tsp, end_tsp = line.strip().split()
            start_tsp = datetime.strptime(start_tsp, "%Y-%m-%d-%H-%M-%S")
            end_tsp = datetime.strptime(end_tsp, "%Y-%m-%d-%H-%M-%S")
            time_slots.append((start_tsp.timestamp(), end_tsp.timestamp()))

    return average_travel_time, time_slots


def generate_sp(new_path, traj_start_day, days_num, roads_dict):
    json_path = new_path + "/Trajs_json/"

    for i in range(traj_start_day, traj_start_day + days_num):
        start = time.time()
        print("Generate SP for Day {}:".format(i))
        trajs = read_json_trajs(json_path, i)
        print("Read trajs finished.")

        sp, tsls = cal_avg_travel_time(trajs, time_slot_duration, roads_dict)
        print("Cal initial sp finished.")
        del trajs

        write_initial_sp_and_timeslots(sp, tsls, new_path, i)
        print("Write initial sp finished.")

        sp, tsls = read_initial_sp_and_timeslots(new_path, i)
        print("read initial sp finished.")

        sp_patched = patch_zero(sp, tsls, roads_dict)
        print("Patch zero finished.")

        write_final_sp(sp_patched, new_path, i)
        print("Write final sp finished.")
        end = time.time()
        print("Cost Time:", end - start, 's')


if __name__ == '__main__':
    start0 = time.time()
    # road_id_mapping, new_road_ids = read_roadMap(bj_traj_path)
    # road_to_speed = read_roadInfo(bj_traj_path)
    # reversed_road_id_mapping = {value: key for key, value in road_id_mapping.items()}
    # roads_dict, two_way_roads_map = read_roadNew(bj_traj_path, new_road_ids, road_to_speed, reversed_road_id_mapping)
    # print("Read map finished.")
    #
    # all_trajs = read_trajs(bj_traj_path, traj_start_day)
    # converted_trajs = convert_road_ids(all_trajs, road_id_mapping, two_way_roads_map)
    # write_trajs(converted_trajs, bj_new_generated_path + "/Trajs_json/")
    # print("Convert trajs finished.")

    roads_dict = read_map(bj_new_generated_path, bj_traj_path)
    print("Read map finished.")
    generate_sp(bj_new_generated_path, traj_start_day, SP_days, roads_dict)

    end0 = time.time()
    print("Generate speed profile finished. Cost Time:", end0 - start0, 's')

