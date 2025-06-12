import os
import numpy as np
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
from pathlib import Path
import re
from matplotlib.ticker import ScalarFormatter

def parse_json_file(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)  # Assuming the JSON file contains a list or array of numbers
    return data

def plot_lr(lr, out_dir):
    iters = list(range(len(lr)))
    # 创建一个新的图形
    plt.figure()
    # 绘制第一条曲线
    plt.plot(iters, lr, label='Global learning rate', color='blue')
    # 添加标题和标签
    plt.legend(fontsize=16)
    # plt.title('Loss curve')
    plt.xlabel('Epochs',fontsize=16)
    plt.ylabel('Global learning rate',fontsize=16)
    # 显示图例
    ax = plt.gca()
    # ax.set_aspect(1.0)
    # 设置纵坐标轴为科学记数法
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style='sci', scilimits=(-3, 4), axis='y')
    ax.tick_params(axis='both', which='major', labelsize=16)
    plt.savefig(f'{out_dir}/Global learning rate.png',dpi=300, bbox_inches='tight')
    # 显示图形
    # plt.show()
    plt.close()


if __name__ == "__main__":
    out_dir = "../../onion_train_data/onion_train_results_APPENDIX"
    file_path = "../../onion_train_data/onion_train_results_APPENDIX/train_results_2A/EXP2A_42_Onion_input_adam_scheduler/train/training_lrs.json"
    lr = parse_json_file(file_path)
    plot_lr(lr, out_dir)