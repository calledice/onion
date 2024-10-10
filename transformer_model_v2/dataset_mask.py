import os
import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def create_embedding_mask(n, m, max_n, max_m):
    """
    生成一个二维掩码矩阵，确保只保留 n×m 的部分，其余部分为 0。
    参数:
    - n (int): 保留的行数。
    - m (int): 保留的列数。
    - max_n (int): 最大行数。
    - max_m (int): 最大列数。
    - device (str): 执行计算的设备 ('cpu' 或 'cuda')。
    返回:
    - mask (Tensor): 形状为 (1, 1, max_n, max_m) 的掩码矩阵，其中 1 表示应关注的位置，0 表示不应关注的位置。
    """
    # 创建一个形状为 (n, m) 的矩阵，其中 1 表示应关注的位置
    ones_matrix = torch.ones(n, m)
    # 将矩阵扩展为 (1, 1, n, m) 的形状
    ones_matrix = ones_matrix
    # 创建一个形状为 (1, 1, max_n, max_m) 的掩码矩阵
    mask = torch.zeros(max_n, max_m)
    # 将保留的部分设置为 1
    mask[:n, :m] = ones_matrix
    return mask

def create_sequence_mask(seq_len, keep_seq_len, num_heads=8):
    """
    创建一个序列掩码，屏蔽除指定序列长度外的所有位置。
    参数:
    - seq_len (int): 序列的总长度。
    - keep_seq_len (int): 需要保留的序列长度。
    - num_heads (int): 注意力头的数量。
    x = 1,100,2048
    多头：1,8,100,256
    score: 1,8,100,100
    mask: 1,8,40,40
    softmax()
    返回:
    - mask (Tensor): 形状为 (1, num_heads, seq_len, seq_len) 的掩码张量。
    """
    # 创建一个全为 0 的掩码张量
    mask = torch.zeros(num_heads, seq_len, seq_len)
    # 将需要保留的部分设置为 1
    mask[:, :keep_seq_len, :keep_seq_len] = 1
    return mask

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
        # 创建一个空的numpy数组，用于存储填充后的结果
        if type(arrays[i]) == np.ndarray:
            if pos_pad_len != None:
                pad_len = pos_pad_len - len(arrays[i])
                pad = np.zeros((pad_len, arrays[i].shape[1]), dtype=np.float32)
                arrays[i] = np.concatenate([arrays[i], pad], axis=0)

            arrays[i] = arrays[i].tolist()  # 在此处将ndarray转成列表，至此arrays列表的任何一个递归子列表都是list类型
        arr = arrays[i]
        if type(arr[0]) != list:  # 当遍历到某个列表的元素不是列表时，则可以认为已经遍历到最内层的列表，因为我们的目标是对最内层列表做填充
            original_length = len(arr)
            arrays[i] = np.pad(arr, (0, target_length - original_length), 'constant')
            original_lengths.append(original_length)  # 保存内层列表原始长度
        else:
            pad_arrays_to_length(arr, target_length)  # 递归调用
        arrays[i] = np.array(arrays[i], dtype=np.float32)

    return arrays, original_lengths


class OnionDataset(Dataset):
    def __init__(self, pth_, max_input_len=100, max_rz_len=10000, num_head = 8):
        '''
        读数据集，做padding操作
        :param pth_: 数据集路径
        :param max_input_len: 最大输入长度，用于做padding
        :param max_rz_len: 最大r*z的长度
        '''
        super().__init__()
        dataset = h5py.File(pth_, 'r')
        x = dataset["x"]
        y = dataset["y"]
        regi = dataset['regi']
        posi = dataset['posi']
        inputs_list = [x[str(i)][:][:-3].flatten() for i in range(len(x))]  # 收集输入数据
        outputs_list = [y[str(i)][:].flatten() for i in range(len(y))]  # 收集输出数据
        regi_list = [regi[str(i)][:] for i in range(len(regi))]  # 收集regi信息
        posi_list = [posi[str(i)][:] for i in range(len(posi))]  # 收集posi信息
        self.max_input_len = max_input_len
        self.max_rz_len = max_rz_len
        self.num_head = num_head
        self.info_list = [x[str(i)][:][-3:].flatten() for i in range(len(x))]  # 收集输入数据
        self.padded_input, self.input_len_org = pad_arrays_to_length(inputs_list, max_input_len)
        self.padded_output, _ = pad_arrays_to_length(outputs_list, max_rz_len)
        self.padded_regi, _ = pad_arrays_to_length(regi_list, max_rz_len)
        self.padded_posi, _ = pad_arrays_to_length(posi_list, max_rz_len, pos_pad_len=max_input_len)

    def __getitem__(self, idx):
        regi_pos_idx, r, z = self.info_list[idx]
        regi_pos_idx, r, z = int(regi_pos_idx), int(r), int(z)
        input_length = self.input_len_org[idx]
        input = torch.tensor(self.padded_input[idx], requires_grad=True, dtype=torch.float32)
        regi = self.padded_regi[regi_pos_idx]
        regi = torch.tensor(regi).unsqueeze(0).repeat(self.max_input_len, 1)

        # regi_old = torch.concat([
        #     torch.concat([torch.tensor(regi[:input_length, :r * z], requires_grad=True, dtype=torch.float32),
        #                   torch.tensor(regi[:input_length, r * z:], requires_grad=False, dtype=torch.float32)
        #                   ##？？？input_length：      对于每个索引的sample  梯度计算的区域是不一样的。可以消融实验对比一下
        #                   ], dim=1),
        #     torch.tensor(regi[input_length:, :], requires_grad=False, dtype=torch.float32)
        #     ], dim=0
        # )
        part1 = torch.concat([
            regi[:input_length, :r * z].clone().detach().requires_grad_(True),
            regi[:input_length, r * z:].clone().detach(),
        ], dim=1)
        # 第二部分，不需要梯度计算
        part2 = regi[input_length:, :].clone().detach()
        # 将两部分拼接起来
        regi = torch.concat([part1, part2], dim=0)
        posi = torch.tensor(np.array(self.padded_posi[regi_pos_idx]))

        # posi_old = torch.concat([
        #     torch.concat([torch.tensor(posi[:input_length, :r * z], requires_grad=True, dtype=torch.float32),
        #                   torch.tensor(posi[:input_length, r * z:], requires_grad=False, dtype=torch.float32)##？？？input_length：      对于每个索引的sample  梯度计算的区域是不一样的。可以消融实验对比一下
        #                   ], dim=1),
        #     torch.tensor(posi[input_length:, :], requires_grad=False, dtype=torch.float32)
        #     ], dim=0
        # )   # 二维padding的梯度屏蔽，首先对前input_length行的前r*z列开放梯度更新，屏蔽前input_length行r*z列之后的所有列的梯度
            # 其次对第input_length列之后的所有内容屏蔽梯度
        part1_ = torch.concat([
            posi[:input_length, :r * z].clone().detach().requires_grad_(True),
            posi[:input_length, r * z:].clone().detach()
        ], dim=1)
        # 构建不需要梯度的第二部分
        part2_ = posi[input_length:, :].clone().detach()
        # 将两部分张量沿着维度0拼接
        posi = torch.concat([part1_, part2_], dim=0)
        output = self.padded_output[idx]
        info = np.array((input_length, r, z))
        embedding_mask = create_embedding_mask(info[0],info[1]*info[2], self.max_input_len, self.max_rz_len)
        sequence_mask = create_sequence_mask(self.max_input_len,info[0],num_heads = self.num_head)
        return (input, regi, posi, info, embedding_mask, sequence_mask), output

    def __len__(self):
        return len(self.padded_input)

