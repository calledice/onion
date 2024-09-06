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

   
class CNN_Base(nn.Module):
    def __init__(self, n, r, z):
        super().__init__()
        self.n = n
        self.r = r
        self.z = z
        self.out_dim = n * r * z * 4
        self.fc1 = nn.Sequential(
            nn.Linear(n, 4096),
            nn.GELU(),
            nn.Linear(4096, n * r * z),
            nn.GELU()
        )
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=n, out_channels=n, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(self.out_dim, r * z),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.fc1(x)
        z = z.reshape(-1, self.n, self.r, self.z)
        z = self.conv(z)
        z = z.reshape(-1, self.out_dim)
        z = self.fc2(z)
        return z

    
def weighted_mse_loss(pred, target, weight=None):
    '''
    加权MSELoss, 对预测错误的负样本施加100倍惩罚, 为了让该学成0的位置学成0
    '''
    # 计算均方误差
    mse_loss = (pred - target) ** 2

    penalty = torch.where((pred == 0) & (target != 0), torch.tensor(100.0), torch.tensor(1.0))

    # 计算损失的平均值
    return (mse_loss * penalty).mean()

def summary():
    n = 43
    r = 17
    z = 25
    flatten_len = r*z
    input_shapes = [(1,n),(1,r,z),(1,n,r,z)]
    inputs = [torch.randn(*shape) for shape in input_shapes]
    onion = Onion(n=n, max_r=r, max_z=z)
    summary(onion,input_data=inputs)

if __name__ =="__main__":
    n = 40
    r = 36
    z = 32
    x = torch.rand(64, n)
    model = CNN_Base(n, r, z)
    model(x)