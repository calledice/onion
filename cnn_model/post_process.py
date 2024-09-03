import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from dataset import OnionDataset
from torch.utils.data import DataLoader
import torch
from onion_model import Onion, Config, ConvEmbModel
import json
import time
import h5py
import os
import glob


def plot_data(data, title, save_path,i):
    # 计算最大值和最小值
    max_val = data.max()
    min_val = data.min()

    # 创建等高线级别
    levels = np.linspace(min_val, max_val, 20)
    plt.figure()
    plt.pcolor(data, cmap='jet')
    plt.colorbar(label='ne')
    plt.title(title)
    ax = plt.gca()
    ax.set_aspect(1.0)
    plt.savefig(save_path + "/" +f"{i}-"+ title)
    plt.show()
    plt.close()


if __name__ == "__main__":
    pred_path = "./output/test/preds.json"
    label_path = "./output/test/labels.json"

    preds = json.load(open(pred_path, 'r'))
    labels = json.load(open(label_path, 'r'))
    title_pred = 'preds'
    title_label = 'labels'
    title_error = 'error'
    save_path = "./output"
    ave_error_list = []
    for i in range(len(preds)):
        error = abs(np.matrix(preds[i]).T-np.matrix(labels[i]).T)/np.max(np.matrix(labels[i]).T)*100
        ave_error_list.append(np.average(error))
        plot_data(np.matrix(preds[i]).T,title_pred,save_path,i)
        plot_data(np.matrix(labels[i]).T,title_label,save_path,i)
        plot_data(error,title_error,save_path,i)
    ave_error_all = np.average(ave_error_list)
    print(f"ave_error_all = {ave_error_all}")
    print("good")