import os
import numpy as np
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
from pathlib import Path
import re
import torch
import h5py
import seaborn as sns 
# 获取当前源程序所在的目录
script_dir = os.path.dirname(os.path.abspath(__file__))

# 切换工作目录到源程序所在的目录
os.chdir(script_dir)
def plot_scatter(input,label2results,ave_label2result_error,title,save_path,i):
    plt.figure()
    # error = np.array(ave_label2result_error) * abs(np.max(input))
    ave = np.average(ave_label2result_error)
    plt.scatter(range(len(input)), input, color='b', marker='^', s=15, alpha=0.8)
    plt.scatter(range(len(label2results)), label2results, color='r', marker='o', s=15, alpha=0.5)
    # plt.errorbar(range(len(input)), input, yerr=error, fmt='none', ecolor='r', capsize=5, alpha=0.5)
    plt.legend(labels=['SXR data','Target profile@BP'])
    # plt.legend(labels=['SXR data',"Reconstuction profile@BP",'Target profile@BP'])
    plt.xlabel('$\t{Channel}$',fontsize=16)
    plt.ylabel('$\t{I}_{SXR}(a.u.)$',fontsize=16)
    plt.title(title,fontsize=16)
    ax = plt.gca()
    # ax.set_aspect(1.0)
    ax.tick_params(axis='both', which='major', labelsize=16)
    plt.savefig(save_path + "/" +f"{ave}-{i}-"+ title+".png",dpi = 300, bbox_inches='tight')
    # plt.show()
    plt.close()

def plot_distribution(data, save_path):
    plt.figure(figsize=(10,  6))
    data = [x for x in data if x < 0.15]
    # 绘制直方图（hist=True）并叠加 KDE 曲线（kde=True）
    ax = sns.histplot( 
        data,
        bins=30,            # 调整 bins 数量 
        kde=True,           # 同时绘制 KDE 
        stat="percent",     # 标准化（使得直方图和 KDE 在同一尺度）
        color="skyblue",    # 直方图颜色 
        edgecolor="black",  # 直方图边框颜色 
        alpha=0.6,         # 透明度 
        line_kws={'lw': 2, 'color': 'red'},  # KDE 线样式 
    )
    
    # 在每个柱子上方标注其值（density）
    for p in ax.patches: 
        if p.get_height()  > 0.05:
            ax.annotate( 
                f"{p.get_height():.1f}",   # 显示3位小数 
                (p.get_x()  + p.get_width()  / 2, p.get_height()),   # 位置（柱子顶部中心）
                ha="center",  # 水平居中 
                va="bottom",  # 垂直对齐底部 
                xytext=(0, 5),  # 文字偏移（5像素）
                textcoords="offset points",  # 相对偏移 
                fontsize=8,  # 字体大小 
            )
    
    # 添加标题和标签 
    plt.title("Error  Distribution: Histogram + KDE", fontsize=14)
    plt.xlabel("Error  Value", fontsize=12)
    plt.ylabel("Density(%)",  fontsize=12)
    plt.grid(linestyle="--",  alpha=0.3)
    
    # 确保保存路径存在 
    os.makedirs(save_path,  exist_ok=True)
    
    # 保存图片（PNG格式，300 DPI）
    plt.savefig( 
        os.path.join(save_path,  "Histogram and KDE.png"), 
        dpi=300,
        bbox_inches="tight",
    )
    
    plt.close()   # 关闭图形，避免内存泄漏

def dataset_evaluation(inputs,label2results,file_path,dataset_name):
    print("start dataset_evaluation")
    save_file = file_path+"eva_data/"
    # 如果目录不存在，则创建（exist_ok=True 避免报错）
    os.makedirs(save_file,  exist_ok=True)
    ave_label2result_error2_list = []# 由label获得的弦积分结果与input的偏差
    error_record_path = save_file+"/"+dataset_name+"_error_record.txt"
    title_data = "SXR data and BPs"
    for i in tqdm(range(len(inputs)), desc='Visualizing'):
        max_input = np.max(np.array(inputs[i])) 
        if max_input == 0 or np.isnan(max_input)  or np.isinf(max_input): 
            print(f"Warning: Invalid max_input at index {i}, skipping...")
            continue  # 跳过无效数据 
        ave_label2result_error2 = abs(np.array(label2results[i])  - np.array(inputs[i]))  / max_input # E2
        # ave_label2result_error2 = abs(np.array(label2results[i])-np.array(inputs[i]))/np.max(np.array(inputs[i]))# E2
        # ave_label2result_error2 = abs(np.array(label2results[i])-np.array(inputs[i]))/np.array(inputs[i])# E2-up
        ave_label2result_error2_list.append(np.average(ave_label2result_error2))
        if i < 30000 or i==383031 or i==183721:
            # max_val = max(np.max(preds[i]),np.max(labels[i]))
            # plot_data(np.matrix(preds[i]),title_pred,save_path,i,max_val)
            # plot_data(np.matrix(labels[i]),title_label,save_path,i,max_val)
            # max_val_error = np.max(relative_error1)
            # plot_data(relative_error1,title_error,save_path,i,max_val_error)
            # plot_scatter(inputs[i],label2results[i],ave_label2result_error2,title_data,save_file,i)
            pass
    ave_label2result_error_all = np.average(ave_label2result_error2_list)
    print(f"最小值：{min(ave_label2result_error2_list)}")
    print(f"最小值索引：{ave_label2result_error2_list.index(min(ave_label2result_error2_list))}")
    print(f"最大值：{max(ave_label2result_error2_list)}")
    print(f"最大值索引：{ave_label2result_error2_list.index(max(ave_label2result_error2_list))}")

    plot_distribution(ave_label2result_error2_list,save_file)

    with open(error_record_path,"a") as file:
        file.write(f"ave_label2result_all = {ave_label2result_error_all}\n")
        file.write(f"最小值：{min(ave_label2result_error2_list)}\n")
        file.write(f"最小值索引：{ave_label2result_error2_list.index(min(ave_label2result_error2_list))}\n")    
        file.write(f"最大值：{max(ave_label2result_error2_list)}\n")
        file.write(f"最大值索引：{ave_label2result_error2_list.index(max(ave_label2result_error2_list))}\n")
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
    target = "HL_2A"#"HL_2A"
    if target == "EAST":
        file_path = "../../onion_train_data/dataset/train_data_East/"
        dataset_name = "EAST_train_database.h5"
    else:
        file_path = "../../onion_train_data/dataset/train_data_HL_2A/"
        dataset_name = "HL_2A_train_database.h5"
    dataset_path = file_path+"data/"+dataset_name

    # 创建Path对象
    # dataset_path0 = file_path+"data/HL_2A_test_database.h5"
    # dataset_path1 = file_path+"data/HL_2A_val_database.h5"
    inputs_list, outputs_list, R_matrix = load_h5(dataset_path)
    # inputs_list0, outputs_list0, R_matrix0 = load_h5(dataset_path0)
    # inputs_list1, outputs_list1, R_matrix1 = load_h5(dataset_path1)
    label2results_list = []
    for i in range(len(outputs_list)):
        label2results_list.append(R_matrix@outputs_list[i])
    
    dataset_evaluation(inputs_list,label2results_list,file_path,dataset_name)

    print("finish")