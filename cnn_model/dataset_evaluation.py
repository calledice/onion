import os
import numpy as np
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
from pathlib import Path
import re
import torch
import h5py
# 获取当前源程序所在的目录
script_dir = os.path.dirname(os.path.abspath(__file__))

# 切换工作目录到源程序所在的目录
os.chdir(script_dir)

def dataset_evaluation(inputs,label2results,save_file,dataset_name):
    print("start dataset_evaluation")
    ave_label2result_error2_list = []# 由label获得的弦积分结果与input的偏差
    error_record_path = save_file+"/"+dataset_name+"_error_record.txt"
    for i in tqdm(range(len(inputs)), desc='Visualizing'):
        ave_label2result_error2 = abs(np.array(label2results[i])-np.array(inputs[i]))/np.max(np.array(inputs[i]))# E2
        # ave_label2result_error2 = abs(np.array(label2results[i])-np.array(inputs[i]))/np.array(inputs[i])# E2-up
        ave_label2result_error2_list.append(np.average(ave_label2result_error2))
    ave_label2result_error_all = np.average(ave_label2result_error2_list)

    with open(error_record_path,"a") as file:
        file.write(f"ave_label2result_all = {ave_label2result_error_all}\n")
    print(f"ave_label2result_all = {ave_label2result_error_all}")
    print("good")


def find_json_files(directory, prefix):
    path = Path(directory)
    pattern = f"{prefix}*.json"
    return sorted(
        path.glob(pattern), 
        key=lambda p: int(re.search(rf'{prefix}(\d+)', p.name).group(1)) if re.search(rf'{prefix}(\d+)', p.name) else -1
    )
def parse_json_file(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)  # Assuming the JSON file contains a list or array of numbers
    return data


def load_h5(pth_):
    with h5py.File(pth_, 'r') as dataset:
        x_group = dataset["x"]
        y_group = dataset["y"]
        R_matrix = dataset['posi']['0'][:]
        inputs_list = [x_group[str(i)][:][:-3].flatten() for i in range(len(x_group))]
        outputs_list = [y_group[str(i)][:] for i in range(len(y_group))]
    
    return inputs_list, outputs_list, R_matrix
if __name__ == "__main__": #评估数据集

    file_path = "../../onion_train_data/phantomdata/"
    dataset_name = "HL-2A_test_database_1_100_1000.h5"
    dataset_path = file_path+dataset_name

    # 创建Path对象
    inputs_list, outputs_list, R_matrix = load_h5(dataset_path)
    label2results_list = []
    for i in range(len(outputs_list)):
        label2results_list.append(R_matrix@outputs_list[i])
    save_file = file_path
    dataset_evaluation(inputs_list,label2results_list,save_file,dataset_name)

    print("finish")