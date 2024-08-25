import os
import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def pad_arrays_to_length(arrays, target_length=101, label=0, pos_pad_len=None):
    """
    将列表中的数组padding到指定长度，并记录原始长度。
    参数:
    - arrays: 不同长度数组的列表
    - target_length: 目标padding长度，默认为101
    - pos_pad_len: 对position matrix的填充大小，
    返回:
    - padded_arrays: 所有数组padding到相同长度后的列表
    - original_lengths: 每个数组的原始长度列表
    """
    padded_arrays = []
    original_lengths = []
    # 将列表转换为numpy数组
    for i in range(len(arrays)):
        arr = arrays[i]
        original_length = len(arr)
        arrays[i] = np.pad(arr, (0, target_length - original_length), 'constant')
        original_lengths.append(original_length)

        arrays[i] = torch.concat([
            torch.tensor(arrays[i][:original_length], dtype=torch.float32, requires_grad=True),
            torch.tensor(arrays[i][original_length:], dtype=torch.float32, requires_grad=False)
        ])

    return arrays, original_lengths


def pad_arrays_to_rz(arr, max_r, max_z):
    original_shape = arr.shape
    r, z = original_shape[-2], original_shape[-1]

    pad_width = [(0, 0)] * (len(original_shape) - 2) + \
                [(0, max_r - r),
                 (0, max_z - z)]
    arr = np.pad(arr, pad_width=pad_width, mode='constant', constant_values=0)
    arr = torch.tensor(arr, dtype=torch.float32)
    if len(original_shape) == 2:
        part1 = torch.concat([
            arr[:r, :z].clone().detach().requires_grad_(True),
            arr[:r, z:].clone().detach(),
        ], dim=1)
        # 第二部分，不需要梯度计算
        part2 = arr[r:, :].clone().detach()
        # 将两部分拼接起来
        arr = torch.concat([part1, part2], dim=0)
    elif len(original_shape) == 3:
        part1_ = torch.concat([
            arr[:, :r, :z].clone().detach().requires_grad_(True),
            arr[:, :r, z:].clone().detach()
        ], dim=2)
        # 构建不需要梯度的第二部分
        part2_ = arr[:, r:, :].clone().detach()
        # 将两部分张量沿着维度0拼接
        arr = torch.concat([part1_, part2_], dim=1)

    return arr

class OnionDataset(Dataset):
    def __init__(self, pth_, max_input_len=100, max_r=100, max_z=100):
        '''
        读数据集，做padding操作
        :param pth_: 数据集路径
        :param max_input_len: 最大输入长度，用于做padding
        :param max_rz_len: 最大r*z的长度
        '''
        super().__init__()
        self.max_r = max_r
        self.max_z = max_z
        dataset = h5py.File(pth_, 'r')
        x = dataset["x"]
        y = dataset["y"]
        regi = dataset['regi']
        posi = dataset['posi']
        inputs_list = [x[str(i)][:][:-3].flatten() for i in range(len(x))]  # 收集输入数据
        self.info_list = [tuple(x[str(i)][:][-3:].flatten()) for i in range(len(x))]  # 收集输入数据
        self.outputs_list = [y[str(i)][:].reshape(int(self.info_list[i][1]), int(self.info_list[i][2])) for i in range(len(y))]  # 收集输出数据
        self.regi_list = [regi[str(i)][:] for i in range(len(regi))]  # 收集regi信息
        self.posi_list = [posi[str(i)][:] for i in range(len(posi))]  # 收集posi信息
        for i, r, z in set(self.info_list):
            i, r, z = int(i), int(r), int(z)
            self.posi_list[i] = self.posi_list[i].reshape(self.posi_list[i].shape[0], r, z)
            self.regi_list[i] = self.regi_list[i].reshape(r, z)

        self.max_input_len = max_input_len
        self.padded_input, self.input_len_org = pad_arrays_to_length(inputs_list, max_input_len)

    def __getitem__(self, idx):
        regi_pos_idx, r, z = self.info_list[idx]
        regi_pos_idx, r, z = int(regi_pos_idx), int(r), int(z)
        input_length = self.input_len_org[idx]
        input = self.padded_input[idx]

        self.padded_output = pad_arrays_to_rz(self.outputs_list[idx], self.max_r, self.max_z)
        regi = pad_arrays_to_rz(self.regi_list[regi_pos_idx], self.max_r, self.max_z)
        posi = pad_arrays_to_rz(self.posi_list[regi_pos_idx], self.max_r, self.max_z)
        pad_shape = (input.size(0) - posi.size(0), posi.size(1), posi.size(2))
        pad = torch.zeros(pad_shape, dtype=torch.float32, requires_grad=False)
        posi = torch.concat([posi, pad], dim=0)

        output = self.padded_output.flatten()
        info = (input_length, r, z)
        return (input, regi, posi, info), output

    def __len__(self):
        return len(self.padded_input)

onion = OnionDataset("../data_Phantom/phantomdata/mini_1_test_database.h5")
onion.__getitem__(90)
