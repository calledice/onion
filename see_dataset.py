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
import torch.nn as nn
import time
import h5py

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ######################################################
    #查看test/train数据
    test_input_path = "./data_Phantom/phantomdata/mini_1_test_database.h5"
    dataset = h5py.File(test_input_path, 'r')
    x = dataset["x"]
    y = dataset["y"]
    regi = dataset['regi']
    posi = dataset['posi']
    inputs_list = [x[dkey][:][:-3].flatten() for dkey in x.keys()]  # 收集输入数据
    outputs_list = [y[dkey][:].flatten() for dkey in y.keys()]
    label = (y["0"][:].reshape(int(x["0"][-2]),int(x["0"][-1]))).T
    # 创建图形和轴
    fig, ax = plt.subplots()
    # 绘制等高线图
    contour = ax.contourf(label, cmap='jet')
    # 设置坐标轴的刻度长度相同
    ax.set_aspect('equal')
    # 添加颜色条
    cbar = fig.colorbar(contour, ax=ax)
    # 显示图形
    plt.show()
    print("good")