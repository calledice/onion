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
        channels = n * 2 + 1

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(num_features=channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(num_features=channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=channels, out_channels=channels * 2, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channels * 2, out_channels=channels * 2, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(num_features=channels * 2),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d(output_size=(10, 10))
        )

        conv_out_dim = channels * 2 * 10 * 10
        fc_out_dim = max_r * max_z

        self.fc = nn.Sequential(
            nn.Linear(in_features=conv_out_dim, out_features=fc_out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(in_features=fc_out_dim, out_features=fc_out_dim),
            nn.Sigmoid()
        )

    def forward(self, input, regi, posi):
        input = self.conv_upsample(input)
        regi = regi.unsqueeze(dim=0).transpose(0, 1)
        final_input = torch.concat([input, regi, posi], dim=1)
        conv_out = self.net(final_input)    
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

# if __name__ =="__main__":
#     n = 23
#     r = 32
#     z = 36
#     flatten_len = r*z
#     input_shapes = [(1,n),(1,r,z),(1,n,r,z)]
#     inputs = [torch.randn(*shape) for shape in input_shapes]
#     onion = Onion(n=n, max_r=r, max_z=z)
#     summary(onion,input_data=inputs)
