from dataset import OnionDataset
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
import math
import json
from torchinfo import summary

class Config:
    def __init__(self, n_layer, n_head, dropout, bias, dtype, batch_size, max_input_len, max_rz_len):
        self.n_layer = n_layer
        self.n_head = n_head
        self.dropout = dropout
        self.bias = bias
        self.dtype = dtype
        self.batch_size = batch_size
        self.max_input_len = max_input_len
        self.max_rz_len = max_rz_len


class ConvEmbModel(nn.Module):
    """
    使用卷积操作将input从(batch_size, max_n) padding到(batch_size, max_n, max_r, max_ z)
    """
    def __init__(self, max_r=100, max_z=100):
        super().__init__()
        self.r = max_r
        self.z = max_z
        out_channels = max_r * max_z
        self.conv = nn.Conv1d(in_channels=1, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv(x.unsqueeze(1)).transpose(1, 2)
        x = x.reshape(x.shape[0], x.shape[1], self.r, self.z)
        return x


class Onion(nn.Module):
    def __init__(self, n=100, max_r=100, max_z=100):
        super(Onion, self).__init__()
        self.conv_upsample = ConvEmbModel(max_r, max_z)
        # channels = n * 2 + 1
        channels = n

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=128, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d(output_size=(2, 2))
        )

        conv_out_dim = 512*2*2
        fc_out_dim = max_r * max_z

        self.fc = nn.Sequential(
            nn.Linear(in_features=conv_out_dim, out_features=fc_out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(in_features=fc_out_dim, out_features=fc_out_dim),
            nn.Sigmoid()
        )

    def forward(self, input):
        input = self.conv_upsample(input)
        conv_out = self.net(input)
        conv_out = conv_out.reshape(conv_out.size(0), -1)
        out = self.fc(conv_out)
        return out
    
def weighted_mse_loss(pred, target, weight=None):
    '''
    加权MSELoss, 对预测错误的负样本施加惩罚, 为了让该学成0的位置学成0
    '''
    # 计算均方误差
    mse_loss = (pred - target) ** 2

    penalty = torch.where((pred == 0) & (target != 0), torch.tensor(100.0), torch.tensor(1.0))

    # 计算损失的平均值
    return (mse_loss * penalty).mean()

if __name__ == "__main__":
    input_shapes = [(1,43)]
    inputs = [torch.randn(*shape) for shape in input_shapes]
    onion = Onion(n=43, max_r=17, max_z=25)
    summary(onion,input_data=inputs)