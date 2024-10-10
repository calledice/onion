import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
# from dataset import OnionDataset
from torch.utils.data import DataLoader
import torch
# from onion_model import Onion, Config, ConvEmbModel
import json
import torch.nn as nn
import time
import h5py

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ######################################################
    #查看test/train数据
    test_input_path = "/onion/data_Phantom/phantomdata/HL-2A_test_database_1_100_1000.h5"
    dataset = h5py.File(test_input_path, 'r')
    x = dataset["x"]
    y = dataset["y"]
    regi = dataset['regi']
    posi = dataset['posi']
    info_list = [tuple(x[str(i)][:][-3:].flatten()) for i in range(len(x))]
    inputs_list = [x[str(i)][:][:-3].flatten() for i in range(len(x))] 
     # 收集输入数据
    print(f"num = {len(inputs_list)}")
    outputs_list = [y[str(i)][:].reshape(int(info_list[i][1]), int(info_list[i][2])) for i in
                             range(len(y))]
    regi_list = [regi[str(i)][:] for i in range(len(regi))]  # 收集regi信息
    posi_list = [posi[str(i)][:] for i in range(len(posi))]
    plt.imshow(np.sum(np.array(posi_list[0]),axis=0).reshape(32,36), cmap='gray', interpolation='nearest')
    # 显示颜色条
    plt.colorbar()
    plt.show()
    r = int(x["0"][-2])
    z = int(x["0"][-1])
    n = len(x["0"])-3
    label = (y["0"][:].reshape(r,z))
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
    print(f"n = {n}")
    print(f"r = {r}")
    print(f"z = {z}")
    print("good")