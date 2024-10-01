import os
import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class OnionDataset(Dataset):
    def __init__(self, pth_):
        '''
        读数据集，做padding操作
        :param pth_: 数据集路径
        :param max_input_len: 最大输入长度，用于做padding
        :param r 
        :param z
        '''
        super().__init__()
        dataset = h5py.File(pth_, 'r')
        x = dataset["x"]
        y = dataset["y"]
        self.posi = dataset['posi']['0'][:]   # 所有posi都相同，只需要第一个
        self.inputs_list = [
            torch.tensor(x[str(i)][:][:-3].flatten(), dtype=torch.float32) for i in range(len(x))
        ]  # 收集输入数据
        self.info_list = [tuple(x[str(i)][:][-3:].flatten()) for i in range(len(x))]  # 收集输入数据
        self.outputs_list = [
            torch.tensor(y[str(i)][:], dtype=torch.float32) for i in range(len(y))
        ]  # 收集输出数据
        # self.posi_list = [posi[str(i)][:] for i in range(len(posi))]  # 收集posi信息

        r = int(self.info_list[0][1])
        z = int(self.info_list[0][2])
        
        # 对posi做归一化操作
        # self.posi_norm = F.normalize(self.posi, p=2)

        self.posi = torch.tensor(self.posi, dtype=torch.float32).reshape(self.posi.shape[0], z, r)
        
    def __getitem__(self, idx):
        return (self.inputs_list[idx], self.posi, self.outputs_list[idx])

    def __len__(self):
        return len(self.inputs_list)

if __name__ == '__main__':
    # 获取当前源程序所在的目录
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 切换工作目录到源程序所在的目录
    os.chdir(script_dir)
    dataset = OnionDataset('../data_Phantom/phantomdata/HL-2A_valid_database_1_100_10.h5')
    print(dataset.outputs_list[0].shape)