import os
import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def pad_arrays_to_length(arrays, target_length=100):
    """
    主要是对input维度做padding，将列表中所有的numpy数组从n维填充到target_length维，同时抑制填充位置的梯度更新。
    (dataset_len, n) -> (dataset_len, max_n)
    参数:
    - arrays: 不同长度数组的列表
    - target_length: 目标padding长度，默认为100

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


def pad_arrays_to_rz(arr, max_z, max_r):
    """
    将region和position的r*z的向量拆成[r, z]的矩阵，分别按行和列对r和z进行padding，并做梯度屏蔽
    (region_num, r*z) -> (region_num, r, z) -> (region_num, max_r, max_z)
    """
    original_shape = arr.shape
    r, z = original_shape[-1], original_shape[-2]

    # 构造要做padding的数量
    if len(original_shape) == 2:
        # 二维，对regi做padding
        pad_width = [(0, max_z - z),(0, max_r - r)]
        arr = np.pad(arr, pad_width=pad_width, mode='constant', constant_values=0)
        arr = torch.tensor(arr, dtype=torch.float32)
        part1 = torch.concat([
            arr[:z, :r].clone().detach().requires_grad_(True),
            arr[:z, r:].clone().detach(),
        ], dim=1)
        # 第二部分，不需要梯度计算
        part2 = arr[z:, :].clone().detach()
        # 将两部分拼接起来
        arr = torch.concat([part1, part2], dim=0)
    elif len(original_shape) == 3:
        # 三维，对posi做padding
        pad_width = [(0, 0),(0, max_z - z),(0, max_r - r)]
        arr = np.pad(arr, pad_width=pad_width, mode='constant', constant_values=0)
        arr = torch.tensor(arr, dtype=torch.float32)
        part1_ = torch.concat([
            arr[:, :z, :r].clone().detach().requires_grad_(True),
            arr[:, :z, r:].clone().detach()
        ], dim=2)
        # 构建不需要梯度的第二部分
        part2_ = arr[:, z:, :].clone().detach()
        # 将两部分张量沿着维度0拼接
        arr = torch.concat([part1_, part2_], dim=1)

    else:
        raise ValueError("arr的维度应该是2维或3维，2维代表regi，3维代表posi")
    return arr


class OnionDataset(Dataset):
    def __init__(self, pth_, max_input_len=100, max_r=100, max_z=100):
        '''
        读数据集，做padding操作
        :param pth_: 数据集路径
        :param max_input_len: 最大输入长度，用于做padding
        :param max_r 
        :param max_z
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
        self.outputs_list = [y[str(i)][:].reshape(int(self.info_list[i][2]), int(self.info_list[i][1])) for i in
                             range(len(y))]  # 收集输出数据
        self.regi_list = [regi[str(i)][:] for i in range(len(regi))]  # 收集regi信息
        self.posi_list = [posi[str(i)][:] for i in range(len(posi))]  # 收集posi信息,100个（40,1152）

        #  2A:
        #  posi[0].reshape(36,32)
        #  regi.reshape(36,32)
        #  east:
        #  label_list[0].reshape(75,50)
        # 临时加的，为了不做padding操作
        max_input_len = len(inputs_list[0])
        self.max_r = int(self.info_list[0][1])
        self.max_z = int(self.info_list[0][2])

        self.max_input_len = max_input_len
        self.padded_input, self.input_len_org = pad_arrays_to_length(inputs_list, max_input_len)

        visited_regi = set()  # 在对regi和posi做padding时候，同一个位置只做一次
        self.padded_output = []

        # 将regi, posi, output向量还原成矩阵，并做padding
        for i in tqdm(range(len(inputs_list)), desc="Padding ... "):
            regi_pos_idx, r, z = self.info_list[i]
            regi_pos_idx, r, z = int(regi_pos_idx), int(r), int(z)
            input = self.padded_input[i]

            # 对output做padding
            output = pad_arrays_to_rz(self.outputs_list[i], self.max_z, self.max_r)
            output = output.flatten()
            self.padded_output.append(output)

            # 对regi和posi做padding
            if regi_pos_idx not in visited_regi:
                visited_regi.add(regi_pos_idx)  # 访问过的下标放到集合中，之后不再做padding
                
                # 对regi做padding
                self.regi_list[regi_pos_idx] = self.regi_list[regi_pos_idx].reshape(z, r)
                self.regi_list[regi_pos_idx] = pad_arrays_to_rz(self.regi_list[regi_pos_idx], self.max_z, self.max_r)

                # 对posi做padding，先还原为三维: (n, r, z)
                self.posi_list[regi_pos_idx] = self.posi_list[regi_pos_idx].reshape(
                    self.posi_list[regi_pos_idx].shape[0], z, r)
                # (n, r, z) -> (n, max_r, max_z)
                posi = pad_arrays_to_rz(self.posi_list[regi_pos_idx], self.max_z, self.max_r)
                # (n, max_r, max_z) -> (max_n, max_r, max_z)
                pad_shape = (input.size(0) - posi.size(0), posi.size(1), posi.size(2))
                pad = torch.zeros(pad_shape, dtype=torch.float32, requires_grad=False)
                self.posi_list[regi_pos_idx] = torch.concat([posi, pad], dim=0)

    def __getitem__(self, idx):
        regi_pos_idx, r, z = self.info_list[idx]
        regi_pos_idx, r, z = int(regi_pos_idx), int(r), int(z)
        input_length = self.input_len_org[idx]
        input = self.padded_input[idx]

        output = self.padded_output[idx].flatten()
        info = (input_length, r, z)
        return (input, self.regi_list[regi_pos_idx], self.posi_list[regi_pos_idx], info), output

    def __len__(self):
        return len(self.padded_input)

if __name__ == '__main__':
    # 获取当前源程序所在的目录
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 切换工作目录到源程序所在的目录
    os.chdir(script_dir)
    dataset = OnionDataset('../data_Phantom/phantomdata/HL-2A_valid_database_1_100_1000.h5')
    print(dataset.padded_output[0].shape)