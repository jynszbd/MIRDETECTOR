import os
import json
###FH6,敏感文件，用作恶意文件标记.NT1:用于恶意IP标记
type_to_index = {
    "NT1": 0,
    "FU1": 1,
    "FU2": 2,
    "FU3": 3,
    "FU4": 4,
    "FU5": 5,
    "FU6": 6,
    "FH1": 7,
    "FH2": 8,
    "FH3": 9,
    "FH4": 10,
    "FH5": 11,
    "FH6": 12,
    "PS1": 13,
    "PS2": 14,
    "PS3": 15,
    "PS4": 16,
    "PS5": 17,
    "PS6": 18,
    "PS7": 19,
    "PB1": 20,
    "PB2": 21,
    "PB3": 22,
    "PB4": 23,
    "PB5": 24,
    "PB6": 25,
    "PB7": 26,
    "PB8": 27,
}
def create_folder_for_file(file_path):
    folder_path = os.path.dirname(file_path)

    # 判断文件夹路径是否存在
    if folder_path and not os.path.exists(folder_path):
        os.makedirs(folder_path)

def read_InitLabels(processFile, fileFile):
    process_label = {}
    file_label = {}
    with open(processFile, 'r') as f:
        for line in f:
            process_id, process_tags = line.strip().split(": ")
            process_label[process_id] = process_tags
            process_label[process_id] = eval(process_label[process_id])
    with open(fileFile, 'r') as f:
        for line in f:
            file_id, file_tags = line.strip().split(": ")
            file_label[file_id] = file_tags
            file_label[file_id] = eval(file_label[file_id])
    return process_label, file_label

import numpy as np



def read_InitLabels(processFile, fileFile):
    process_label = {}
    file_label = {}
    with open(processFile, 'r') as f:
        for line in f:
            process_id, process_tags = line.strip().split(": ")
            process_label[process_id] = process_tags
            process_label[process_id] = eval(process_label[process_id])
    with open(fileFile, 'r') as f:
        for line in f:
            file_id, file_tags = line.strip().split(": ")
            file_label[file_id] = file_tags
            file_label[file_id] = eval(file_label[file_id])
    return process_label, file_label




def writeInitLabel_vec(writename, dict):
    with open(writename, 'a') as wf:
        for dictkeys in dict:
            new_line = str(dictkeys) + " "+" ".join(str(x) for x in dict[dictkeys])
            wf.writelines(new_line)
            wf.writelines("\n")

def increase_type_count(vec, type_name, count=1):
    if type_name in type_to_index:
        index = type_to_index[type_name]  # 获取类型对应的索引
        vec[index] += count  # 在向量的对应位置增加计数
    else:
         print(f"未知的类型: {type_name}")

    return vec



def readInitVec(file):
    Init_vec = {}
    process_Init_vec = {}
    with open(file, 'r') as file:
        for line in file:
            # 分割ID和向量部分
            items = line.strip().split(' ')
            node_id = int(items[0])
            vector = np.array([float(x) for x in items[1:]])
            Init_vec[node_id] = vector

    return Init_vec



def InitLabels_Vec(dict):
    entity_vec = np.zeros(28)
    vec_toIndex = {}
    for keys in dict:
        # tags = eval(dict[keys])
        for item in dict[keys]:
            entity_vec = increase_type_count(entity_vec, item)
        vec_toIndex[keys] = entity_vec
        entity_vec = np.zeros(28)
    return vec_toIndex