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
from contextlib import redirect_stdout

class Config:
    def __init__(self, train_path, val_path, test_path, out_dir, with_PI, addloss, randomnumseed, n_layer=None,
            n_head=None, dropout=None, bias=True, dtype=torch.float32, batch_size=256,
            max_n=100, max_r=100, max_z=100, lr=0.0001, epochs=50, early_stop=-1,lambda_l2 = 0.0001,p=2,device_num="0",
            alfa = 0.618, scheduler=False,Module=None):
        self.n_layer = n_layer
        self.n_head = n_head
        self.dropout = dropout
        self.bias = bias
        self.dtype = dtype
        self.batch_size = batch_size
        self.max_n = max_n
        self.max_r = max_r
        self.max_z = max_z
        self.lr = lr
        self.epochs = epochs
        self.early_stop = early_stop
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device('cuda')
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.out_dir = out_dir
        self.with_PI = with_PI
        self.addloss = addloss
        self.lambda_l2 = lambda_l2
        self.randomnumseed = randomnumseed
        self.p = p
        self.device_num = device_num
        self.alfa = alfa
        self.scheduler = scheduler
        self.Module = Module

    def as_dict(self):
        data = self.__dict__.copy()  # 动态捕获所有属性
        return data
        # return {
        #     "train_path": self.train_path,
        #     "val_path": self.val_path,
        #     "test_path": self.test_path,
        #     "out_dir": self.out_dir,
        #     "with_PI": self.with_PI,
        #     "addloss": self.addloss,
        #     "epochs": self.epochs,
        #     "batch_size": self.batch_size,
        #     "lambda_l2": self.lambda_l2,
        #     "p": self.p,
        #     "lr": self.lr,
        #     "device_num": self.device_num,
        #     "alfa": self.alfa
        # }

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
        x = x.reshape(x.shape[0], x.shape[1], self.z, self.r)
        return x

class ResidualBasic(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        residual = x
        out = self.features(x)
        out = out + residual
        out = self.relu(out)
        return out

class Residualscale(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.feature_r = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.feature_x = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=2, padding=0),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x_scale = self.feature_x(x)
        r = self.feature_r(x)
        x = r + x_scale
        x = self.relu(x)
        return x

# class Onion_PI(nn.Module):
#     def __init__(self, n=100, max_r=100, max_z=100):
#         super(Onion_PI, self).__init__()
#         self.conv_upsample = ConvEmbModel(max_r, max_z)
#         channels = n * 2

#         self.net = nn.Sequential(
#             nn.Conv2d(in_channels=channels, out_channels=2*channels, kernel_size=(3, 3), padding=(1, 1)),
#             nn.BatchNorm2d(num_features=2 * channels),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=2*channels, out_channels=2*channels, kernel_size=(3, 3), padding=(1, 1)),
#             nn.BatchNorm2d(num_features=2*channels),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2,stride=2),

#             nn.Conv2d(in_channels=2*channels, out_channels=4*channels, kernel_size=(3, 3), padding=(1, 1)),
#             nn.BatchNorm2d(num_features=4 * channels),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=4*channels, out_channels=4*channels, kernel_size=(3, 3), padding=(1, 1)),
#             nn.BatchNorm2d(num_features=4 * channels),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=4*channels, out_channels=4*channels, kernel_size=(3, 3), padding=(1, 1)),
#             nn.BatchNorm2d(num_features=4*channels),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2,stride=2),

#             nn.Conv2d(in_channels=4*channels, out_channels=8*channels, kernel_size=(3, 3), padding=(1, 1)),
#             nn.BatchNorm2d(num_features=8 * channels),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=8*channels, out_channels=8*channels, kernel_size=(3, 3), padding=(1, 1)),
#             nn.BatchNorm2d(num_features=8 * channels),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=8*channels, out_channels=8*channels, kernel_size=(3, 3), padding=(1, 1)),
#             nn.BatchNorm2d(num_features=8*channels),
#             nn.ReLU(),
#             nn.AdaptiveMaxPool2d(output_size=(3, 3))
#         )

#         conv_out_dim = 8*channels*3*3
#         fc_out_dim = max_r * max_z

#         self.fc = nn.Sequential(
#             nn.Linear(in_features=conv_out_dim, out_features=fc_out_dim),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(in_features=fc_out_dim, out_features=fc_out_dim),
#             nn.ReLU()
#         )

#     def forward(self, input, posi):
#         input = self.conv_upsample(input)
#         final_input = torch.concat([input, posi], dim=1)
#         conv_out = self.net(final_input)
#         conv_out = conv_out.reshape(conv_out.size(0), -1)
#         out = self.fc(conv_out)
#         return out

# class Onion_PI_up(nn.Module):
#     def __init__(self, n=100, max_r=100, max_z=100):
#         super(Onion_PI_up, self).__init__()
#         self.conv_upsample = ConvEmbModel(max_r, max_z)
#         channels = n
#         channels_pi = n

#         self.net1 = nn.Sequential(
#             nn.Conv2d(in_channels=channels, out_channels=2*channels, kernel_size=(3, 3), padding=(1, 1)),
#             nn.BatchNorm2d(num_features=2 * channels),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=2*channels, out_channels=2*channels, kernel_size=(3, 3), padding=(1, 1)),
#             nn.BatchNorm2d(num_features=2*channels),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2,stride=2),

#             nn.Conv2d(in_channels=2*channels, out_channels=4*channels, kernel_size=(3, 3), padding=(1, 1)),
#             nn.BatchNorm2d(num_features=4 * channels),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=4*channels, out_channels=4*channels, kernel_size=(3, 3), padding=(1, 1)),
#             nn.BatchNorm2d(num_features=4 * channels),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=4*channels, out_channels=4*channels, kernel_size=(3, 3), padding=(1, 1)),
#             nn.BatchNorm2d(num_features=4*channels),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2,stride=2),

#             nn.Conv2d(in_channels=4*channels, out_channels=8*channels, kernel_size=(3, 3), padding=(1, 1)),
#             nn.BatchNorm2d(num_features=8 * channels),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=8*channels, out_channels=8*channels, kernel_size=(3, 3), padding=(1, 1)),
#             nn.BatchNorm2d(num_features=8 * channels),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=8*channels, out_channels=8*channels, kernel_size=(3, 3), padding=(1, 1)),
#             nn.BatchNorm2d(num_features=8*channels),
#             nn.ReLU(),
#             nn.AdaptiveMaxPool2d(output_size=(3, 3))
#         )
#         self.net2 = nn.Sequential(
#             nn.Conv2d(in_channels=channels_pi, out_channels=2 * channels_pi, kernel_size=(3, 3), padding=(1, 1)),
#             nn.BatchNorm2d(num_features=2 * channels_pi),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=2 * channels_pi, out_channels=2 * channels_pi, kernel_size=(3, 3), padding=(1, 1)),
#             nn.BatchNorm2d(num_features=2 * channels_pi),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),

#             nn.Conv2d(in_channels=2 * channels_pi, out_channels=4 * channels_pi, kernel_size=(3, 3), padding=(1, 1)),
#             nn.BatchNorm2d(num_features=4 * channels_pi),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=4 * channels_pi, out_channels=4 * channels_pi, kernel_size=(3, 3), padding=(1, 1)),
#             nn.BatchNorm2d(num_features=4 * channels_pi),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),

#             nn.Conv2d(in_channels=4 * channels_pi, out_channels=8 * channels_pi, kernel_size=(3, 3), padding=(1, 1)),
#             nn.BatchNorm2d(num_features=8 * channels_pi),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=8 * channels_pi, out_channels=8 * channels_pi, kernel_size=(3, 3), padding=(1, 1)),
#             nn.BatchNorm2d(num_features=8 * channels_pi),
#             nn.ReLU(),
#             nn.AdaptiveMaxPool2d(output_size=(3, 3))
#         )

#         conv_out_dim = 8*channels*3*3 + 8*channels_pi*3*3
#         fc_out_dim = max_r * max_z

#         self.fc = nn.Sequential(
#             nn.Linear(in_features=conv_out_dim, out_features=fc_out_dim),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(in_features=fc_out_dim, out_features=fc_out_dim),
#             nn.ReLU()
#         )

#     def forward(self, input, posi):
#         input = self.conv_upsample(input)
#         pi = posi
#         conv_in = self.net1(input)
#         conv_pi = self.net2(pi)
#         conv_out = torch.concat([conv_in, conv_pi], dim=1)
#         conv_out = conv_out.reshape(conv_out.size(0), -1)
#         out = self.fc(conv_out)
#         return out

# class Onion_PI_upadd(nn.Module):
#     def __init__(self, n=100, max_r=100, max_z=100):
#         super(Onion_PI_upadd, self).__init__()
#         self.conv_upsample = ConvEmbModel(max_r, max_z)
#         channels = n
#         channels_pi = n

#         self.net1 = nn.Sequential(
#             nn.Conv2d(in_channels=channels, out_channels=2*channels, kernel_size=(3, 3), padding=(1, 1)),
#             nn.BatchNorm2d(num_features=2 * channels),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=2*channels, out_channels=2*channels, kernel_size=(3, 3), padding=(1, 1)),
#             nn.BatchNorm2d(num_features=2*channels),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2,stride=2),

#             nn.Conv2d(in_channels=2*channels, out_channels=4*channels, kernel_size=(3, 3), padding=(1, 1)),
#             nn.BatchNorm2d(num_features=4 * channels),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=4*channels, out_channels=4*channels, kernel_size=(3, 3), padding=(1, 1)),
#             nn.BatchNorm2d(num_features=4 * channels),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=4*channels, out_channels=4*channels, kernel_size=(3, 3), padding=(1, 1)),
#             nn.BatchNorm2d(num_features=4*channels),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2,stride=2),

#             nn.Conv2d(in_channels=4*channels, out_channels=8*channels, kernel_size=(3, 3), padding=(1, 1)),
#             nn.BatchNorm2d(num_features=8 * channels),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=8*channels, out_channels=8*channels, kernel_size=(3, 3), padding=(1, 1)),
#             nn.BatchNorm2d(num_features=8 * channels),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=8*channels, out_channels=8*channels, kernel_size=(3, 3), padding=(1, 1)),
#             nn.BatchNorm2d(num_features=8*channels),
#             nn.ReLU(),
#             nn.AdaptiveMaxPool2d(output_size=(3, 3))
#         )
#         self.net2 = nn.Sequential(
#             nn.Conv2d(in_channels=channels_pi, out_channels=2 * channels_pi, kernel_size=(3, 3), padding=(1, 1)),
#             nn.BatchNorm2d(num_features=2 * channels_pi),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=2 * channels_pi, out_channels=2 * channels_pi, kernel_size=(3, 3), padding=(1, 1)),
#             nn.BatchNorm2d(num_features=2 * channels_pi),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),

#             nn.Conv2d(in_channels=2 * channels_pi, out_channels=4 * channels_pi, kernel_size=(3, 3), padding=(1, 1)),
#             nn.BatchNorm2d(num_features=4 * channels_pi),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=4 * channels_pi, out_channels=4 * channels_pi, kernel_size=(3, 3), padding=(1, 1)),
#             nn.BatchNorm2d(num_features=4 * channels_pi),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),

#             nn.Conv2d(in_channels=4 * channels_pi, out_channels=8 * channels_pi, kernel_size=(3, 3), padding=(1, 1)),
#             nn.BatchNorm2d(num_features=8 * channels_pi),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=8 * channels_pi, out_channels=8 * channels_pi, kernel_size=(3, 3), padding=(1, 1)),
#             nn.BatchNorm2d(num_features=8 * channels_pi),
#             nn.ReLU(),
#             nn.AdaptiveMaxPool2d(output_size=(3, 3))
#         )

#         conv_out_dim = 8*channels_pi*3*3
#         fc_out_dim = max_r * max_z

#         self.fc = nn.Sequential(
#             nn.Linear(in_features=conv_out_dim, out_features=fc_out_dim),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(in_features=fc_out_dim, out_features=fc_out_dim),
#             nn.ReLU()
#         )

#     def forward(self, input, posi):
#         input = self.conv_upsample(input)
#         pi = posi
#         conv_in = self.net1(input)
#         conv_pi = self.net2(pi)
#         conv_out = conv_in+conv_pi
#         conv_out = conv_out.reshape(conv_out.size(0), -1)
#         out = self.fc(conv_out)
#         return out

class Onion_PI_uptime(nn.Module):
    def __init__(self, n=100, max_r=100, max_z=100):
        super(Onion_PI_uptime, self).__init__()
        self.conv_upsample = ConvEmbModel(max_r, max_z)
        channels = n
        channels_pi = n

        self.net1 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=2*channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(num_features=2 * channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=2*channels, out_channels=2*channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(num_features=2*channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(in_channels=2*channels, out_channels=4*channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(num_features=4 * channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=4*channels, out_channels=4*channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(num_features=4 * channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=4*channels, out_channels=4*channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(num_features=4*channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(in_channels=4*channels, out_channels=8*channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(num_features=8 * channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=8*channels, out_channels=8*channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(num_features=8 * channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=8*channels, out_channels=8*channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(num_features=8*channels),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(output_size=(3, 3))
        )
        self.net2 = nn.Sequential(
            nn.Conv2d(in_channels=channels_pi, out_channels=2 * channels_pi, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(num_features=2 * channels_pi),
            nn.ReLU(),
            nn.Conv2d(in_channels=2 * channels_pi, out_channels=2 * channels_pi, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(num_features=2 * channels_pi),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=2 * channels_pi, out_channels=4 * channels_pi, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(num_features=4 * channels_pi),
            nn.ReLU(),
            nn.Conv2d(in_channels=4 * channels_pi, out_channels=4 * channels_pi, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(num_features=4 * channels_pi),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=4 * channels_pi, out_channels=8 * channels_pi, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(num_features=8 * channels_pi),
            nn.ReLU(),
            nn.Conv2d(in_channels=8 * channels_pi, out_channels=8 * channels_pi, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(num_features=8 * channels_pi),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(output_size=(3, 3))
        )

        conv_out_dim = 8*channels_pi*3*3
        fc_out_dim = max_r * max_z

        self.fc = nn.Sequential(
            nn.Linear(in_features=conv_out_dim, out_features=fc_out_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=fc_out_dim, out_features=fc_out_dim),
            nn.ReLU()
        )

    def forward(self, input, posi):
        input = self.conv_upsample(input)
        pi = posi
        conv_in = self.net1(input)
        conv_pi = self.net2(pi)
        conv_out = conv_in*conv_pi
        conv_out = conv_out.reshape(conv_out.size(0), -1)
        out = self.fc(conv_out)
        return out

class Onion_PI_uptime_softplus(nn.Module):
    def __init__(self, n=100, max_r=100, max_z=100):
        super(Onion_PI_uptime_softplus, self).__init__()
        self.conv_upsample = ConvEmbModel(max_r, max_z)
        channels = n
        channels_pi = n

        self.net1 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=2*channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(num_features=2 * channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=2*channels, out_channels=2*channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(num_features=2*channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(in_channels=2*channels, out_channels=4*channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(num_features=4 * channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=4*channels, out_channels=4*channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(num_features=4 * channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=4*channels, out_channels=4*channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(num_features=4*channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(in_channels=4*channels, out_channels=8*channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(num_features=8 * channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=8*channels, out_channels=8*channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(num_features=8 * channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=8*channels, out_channels=8*channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(num_features=8*channels),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(output_size=(3, 3))
        )
        self.net2 = nn.Sequential(
            nn.Conv2d(in_channels=channels_pi, out_channels=2 * channels_pi, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(num_features=2 * channels_pi),
            nn.ReLU(),
            nn.Conv2d(in_channels=2 * channels_pi, out_channels=2 * channels_pi, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(num_features=2 * channels_pi),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=2 * channels_pi, out_channels=4 * channels_pi, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(num_features=4 * channels_pi),
            nn.ReLU(),
            nn.Conv2d(in_channels=4 * channels_pi, out_channels=4 * channels_pi, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(num_features=4 * channels_pi),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=4 * channels_pi, out_channels=8 * channels_pi, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(num_features=8 * channels_pi),
            nn.ReLU(),
            nn.Conv2d(in_channels=8 * channels_pi, out_channels=8 * channels_pi, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(num_features=8 * channels_pi),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(output_size=(3, 3))
        )

        conv_out_dim = 8*channels_pi*3*3
        fc_out_dim = max_r * max_z

        self.fc = nn.Sequential(
            nn.Linear(in_features=conv_out_dim, out_features=fc_out_dim),
            nn.Softplus(),
            nn.Dropout(0.5),
            nn.Linear(in_features=fc_out_dim, out_features=fc_out_dim),
            nn.Softplus()
        )

    def forward(self, input, posi):
        input = self.conv_upsample(input)
        pi = posi
        conv_in = self.net1(input)
        conv_pi = self.net2(pi)
        conv_out = conv_in*conv_pi
        conv_out = conv_out.reshape(conv_out.size(0), -1)
        out = self.fc(conv_out)
        return out

# class Onion_PI_softplus(nn.Module):
#     def __init__(self, n=100, max_r=100, max_z=100):
#         super(Onion_PI_softplus, self).__init__()
#         self.conv_upsample = ConvEmbModel(max_r, max_z)
#         channels = n * 2

#         self.net = nn.Sequential(
#             nn.Conv2d(in_channels=channels, out_channels=2 * channels, kernel_size=(3, 3), padding=(1, 1)),
#             nn.BatchNorm2d(num_features=2 * channels),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=2 * channels, out_channels=2 * channels, kernel_size=(3, 3), padding=(1, 1)),
#             nn.BatchNorm2d(num_features=2 * channels),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),

#             nn.Conv2d(in_channels=2 * channels, out_channels=4 * channels, kernel_size=(3, 3), padding=(1, 1)),
#             nn.BatchNorm2d(num_features=4 * channels),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=4 * channels, out_channels=4 * channels, kernel_size=(3, 3), padding=(1, 1)),
#             nn.BatchNorm2d(num_features=4 * channels),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=4 * channels, out_channels=4 * channels, kernel_size=(3, 3), padding=(1, 1)),
#             nn.BatchNorm2d(num_features=4 * channels),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),

#             nn.Conv2d(in_channels=4 * channels, out_channels=8 * channels, kernel_size=(3, 3), padding=(1, 1)),
#             nn.BatchNorm2d(num_features=8 * channels),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=8 * channels, out_channels=8 * channels, kernel_size=(3, 3), padding=(1, 1)),
#             nn.BatchNorm2d(num_features=8 * channels),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=8 * channels, out_channels=8 * channels, kernel_size=(3, 3), padding=(1, 1)),
#             nn.BatchNorm2d(num_features=8 * channels),
#             nn.ReLU(),
#             nn.AdaptiveMaxPool2d(output_size=(3, 3))
#         )

#         conv_out_dim = 8*channels*3*3
#         fc_out_dim = max_r * max_z

#         self.fc = nn.Sequential(
#             nn.Linear(in_features=conv_out_dim, out_features=fc_out_dim),
#             nn.Softplus(),
#             nn.Dropout(0.5),
#             nn.Linear(in_features=fc_out_dim, out_features=fc_out_dim),
#             nn.Softplus()
#         )

#     def forward(self, input, posi):
#         input = self.conv_upsample(input)
#         final_input = torch.concat([input, posi], dim=1)
#         conv_out = self.net(final_input)
#         conv_out = conv_out.reshape(conv_out.size(0), -1)
#         out = self.fc(conv_out)
#         return out
    
class Onion_input(nn.Module):
    def __init__(self, n=100, max_r=100, max_z=100):
        super(Onion_input, self).__init__()
        self.conv_upsample = ConvEmbModel(max_r, max_z)
        channels = n

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=2 * channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(num_features=2 * channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=2 * channels, out_channels=2 * channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(num_features=2 * channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=2 * channels, out_channels=4 * channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(num_features=4 * channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=4 * channels, out_channels=4 * channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(num_features=4 * channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=4 * channels, out_channels=4 * channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(num_features=4 * channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=4 * channels, out_channels=8 * channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(num_features=8 * channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=8 * channels, out_channels=8 * channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(num_features=8 * channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=8 * channels, out_channels=8 * channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(num_features=8 * channels),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(output_size=(3, 3))
        )

        conv_out_dim = 8*channels*3*3
        fc_out_dim = max_r * max_z

        self.fc = nn.Sequential(
            nn.Linear(in_features=conv_out_dim, out_features=fc_out_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=fc_out_dim, out_features=fc_out_dim),
            nn.ReLU()
        )

    def forward(self, input):
        input = self.conv_upsample(input)
        conv_out = self.net(input)
        conv_out = conv_out.reshape(conv_out.size(0), -1)
        out = self.fc(conv_out)
        return out
    
class Onion_input_softplus(nn.Module):
    def __init__(self, n=100, max_r=100, max_z=100):
        super(Onion_input_softplus, self).__init__()
        self.conv_upsample = ConvEmbModel(max_r, max_z)
        channels = n

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=2 * channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(num_features=2 * channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=2 * channels, out_channels=2 * channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(num_features=2 * channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=2 * channels, out_channels=4 * channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(num_features=4 * channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=4 * channels, out_channels=4 * channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(num_features=4 * channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=4 * channels, out_channels=4 * channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(num_features=4 * channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=4 * channels, out_channels=8 * channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(num_features=8 * channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=8 * channels, out_channels=8 * channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(num_features=8 * channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=8 * channels, out_channels=8 * channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(num_features=8 * channels),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(output_size=(3, 3))
        )

        conv_out_dim = 8*channels*3*3
        fc_out_dim = max_r * max_z

        self.fc = nn.Sequential(
            nn.Linear(in_features=conv_out_dim, out_features=fc_out_dim),
            nn.Softplus(),
            nn.Dropout(0.5),
            nn.Linear(in_features=fc_out_dim, out_features=fc_out_dim),
            nn.Softplus()
        )

    def forward(self, input):
        input = self.conv_upsample(input)
        conv_out = self.net(input)
        conv_out = conv_out.reshape(conv_out.size(0), -1)
        out = self.fc(conv_out)
        return out

# class ResOnion_PI(nn.Module):
#     def __init__(self, n=100, max_r=100, max_z=100):
#         super(ResOnion_PI, self).__init__()
#         self.conv_upsample = ConvEmbModel(max_r, max_z)
#         channels = n * 2
#         self.res1 = nn.ModuleList([ResidualBasic(channels) for _ in range(2)])
#         self.res2scale = Residualscale(channels, 2 * channels)
#         self.res2 = nn.ModuleList([ResidualBasic(2 * channels) for _ in range(1)])
#         self.res3scale = Residualscale(2 * channels, 4 * channels)
#         self.res3 = nn.ModuleList([ResidualBasic(4 * channels) for _ in range(1)])
#         self.res4scale = Residualscale(4 * channels, 8 * channels)
#         self.res4 = nn.ModuleList([ResidualBasic(8 * channels) for _ in range(1)])

#         final_r = math.ceil(math.ceil(math.ceil(max_r / 2) / 2) / 2)
#         final_z = math.ceil(math.ceil(math.ceil(max_z / 2) / 2) / 2)
#         fc_in_dim = 8 * channels * final_r * final_z
#         fc_out_dim = max_r * max_z
#         self.flatten = nn.Flatten(start_dim=1)
#         self.fc = nn.Sequential(
#             nn.Linear(in_features=fc_in_dim, out_features=fc_out_dim),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(in_features=fc_out_dim, out_features=fc_out_dim),
#             nn.ReLU()
#         )

#     def forward(self, input, posi):
#         input = self.conv_upsample(input)
#         z = torch.concat([input, posi], dim=1)
#         for net in self.res1:
#             z = net(z)
#         z = self.res2scale(z)
#         for net in self.res2:
#             z = net(z)
#         z = self.res3scale(z)
#         for net in self.res3:
#             z = net(z)
#         z = self.res4scale(z)
#         for net in self.res4:
#             z = net(z)
#         z = z.reshape(z.size(0), -1)
#         z = self.fc(z)
#         return z

# class ResOnion_PI_up(nn.Module):
#     def __init__(self, n=100, max_r=100, max_z=100):
#         super(ResOnion_PI_up, self).__init__()
#         self.conv_upsample = ConvEmbModel(max_r, max_z)
#         channels = n
#         self.res1 = nn.ModuleList([ResidualBasic(channels) for _ in range(2)])
#         self.res2scale = Residualscale(channels, 2 * channels)
#         self.res2 = nn.ModuleList([ResidualBasic(2 * channels) for _ in range(1)])
#         self.res3scale = Residualscale(2 * channels, 4 * channels)
#         self.res3 = nn.ModuleList([ResidualBasic(4 * channels) for _ in range(1)])
#         self.res4scale = Residualscale(4 * channels, 8 * channels)
#         self.res4 = nn.ModuleList([ResidualBasic(8 * channels) for _ in range(1)])
#         self.admaxpool = nn.AdaptiveMaxPool2d((3,3))

#         final_r = math.ceil(math.ceil(math.ceil(max_r / 2) / 2) / 2)
#         final_z = math.ceil(math.ceil(math.ceil(max_z / 2) / 2) / 2)

#         self.flatten = nn.Flatten(start_dim=1)

#         channels_pi = n
#         self.net2 = nn.Sequential(
#             nn.Conv2d(in_channels=channels_pi, out_channels=2 * channels_pi, kernel_size=(3, 3), padding=(1, 1)),
#             nn.BatchNorm2d(num_features=2 * channels_pi),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=2 * channels_pi, out_channels=2 * channels_pi, kernel_size=(3, 3), padding=(1, 1)),
#             nn.BatchNorm2d(num_features=2 * channels_pi),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),

#             nn.Conv2d(in_channels=2 * channels_pi, out_channels=4 * channels_pi, kernel_size=(3, 3), padding=(1, 1)),
#             nn.BatchNorm2d(num_features=4 * channels_pi),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=4 * channels_pi, out_channels=4 * channels_pi, kernel_size=(3, 3), padding=(1, 1)),
#             nn.BatchNorm2d(num_features=4 * channels_pi),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),

#             nn.Conv2d(in_channels=4 * channels_pi, out_channels=8 * channels_pi, kernel_size=(3, 3), padding=(1, 1)),
#             nn.BatchNorm2d(num_features=8 * channels_pi),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=8 * channels_pi, out_channels=8 * channels_pi, kernel_size=(3, 3), padding=(1, 1)),
#             nn.BatchNorm2d(num_features=8 * channels_pi),
#             nn.ReLU(),
#             nn.AdaptiveMaxPool2d(output_size=(3, 3))
#         )
#         fc_in_dim = 8 * channels * 3 * 3 + 8 * channels_pi * 3 * 3   # (40,3,3)  (41,3,3)   (81,3,3)
#         fc_out_dim = max_r * max_z
#         self.fc = nn.Sequential(
#             nn.Linear(in_features=fc_in_dim, out_features=fc_out_dim),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(in_features=fc_out_dim, out_features=fc_out_dim),
#             nn.ReLU()
#         )
#     def forward(self, input, posi):
#         z = self.conv_upsample(input)
#         pi = posi

#         conv_pi = self.net2(pi)

#         for net in self.res1:
#             z = net(z)
#         z = self.res2scale(z)
#         for net in self.res2:
#             z = net(z)
#         z = self.res3scale(z)
#         for net in self.res3:
#             z = net(z)
#         z = self.res4scale(z)
#         for net in self.res4:
#             z = net(z)

#         z = self.admaxpool(z)

#         conv_out = torch.concat([z, conv_pi], dim=1)
#         conv_out = conv_out.reshape(conv_out.size(0), -1)

#         conv_out = self.fc(conv_out)
#         return conv_out

# class ResOnion_PI_upadd(nn.Module):
#     def __init__(self, n=100, max_r=100, max_z=100):
#         super(ResOnion_PI_upadd, self).__init__()
#         self.conv_upsample = ConvEmbModel(max_r, max_z)
#         channels = n
#         self.res1 = nn.ModuleList([ResidualBasic(channels) for _ in range(2)])
#         self.res2scale = Residualscale(channels, 2 * channels)
#         self.res2 = nn.ModuleList([ResidualBasic(2 * channels) for _ in range(1)])
#         self.res3scale = Residualscale(2 * channels, 4 * channels)
#         self.res3 = nn.ModuleList([ResidualBasic(4 * channels) for _ in range(1)])
#         self.res4scale = Residualscale(4 * channels, 8 * channels)
#         self.res4 = nn.ModuleList([ResidualBasic(8 * channels) for _ in range(1)])
#         self.admaxpool = nn.AdaptiveMaxPool2d((3,3))

#         final_r = math.ceil(math.ceil(math.ceil(max_r / 2) / 2) / 2)
#         final_z = math.ceil(math.ceil(math.ceil(max_z / 2) / 2) / 2)

#         self.flatten = nn.Flatten(start_dim=1)

#         channels_pi = n
#         self.net2 = nn.Sequential(
#             nn.Conv2d(in_channels=channels_pi, out_channels=2 * channels_pi, kernel_size=(3, 3), padding=(1, 1)),
#             nn.BatchNorm2d(num_features=2 * channels_pi),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=2 * channels_pi, out_channels=2 * channels_pi, kernel_size=(3, 3), padding=(1, 1)),
#             nn.BatchNorm2d(num_features=2 * channels_pi),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),

#             nn.Conv2d(in_channels=2 * channels_pi, out_channels=4 * channels_pi, kernel_size=(3, 3), padding=(1, 1)),
#             nn.BatchNorm2d(num_features=4 * channels_pi),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=4 * channels_pi, out_channels=4 * channels_pi, kernel_size=(3, 3), padding=(1, 1)),
#             nn.BatchNorm2d(num_features=4 * channels_pi),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),

#             nn.Conv2d(in_channels=4 * channels_pi, out_channels=8 * channels_pi, kernel_size=(3, 3), padding=(1, 1)),
#             nn.BatchNorm2d(num_features=8 * channels_pi),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=8 * channels_pi, out_channels=8 * channels_pi, kernel_size=(3, 3), padding=(1, 1)),
#             nn.BatchNorm2d(num_features=8 * channels_pi),
#             nn.ReLU(),
#             nn.AdaptiveMaxPool2d(output_size=(3, 3))
#         )
#         fc_in_dim = 8 * channels_pi * 3 * 3   # (40,3,3)  (41,3,3)   (81,3,3)
#         fc_out_dim = max_r * max_z
#         self.fc = nn.Sequential(
#             nn.Linear(in_features=fc_in_dim, out_features=fc_out_dim),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(in_features=fc_out_dim, out_features=fc_out_dim),
#             nn.ReLU()
#         )
#     def forward(self, input, posi):
#         z = self.conv_upsample(input)
#         pi = posi

#         conv_pi = self.net2(pi)

#         for net in self.res1:
#             z = net(z)
#         z = self.res2scale(z)
#         for net in self.res2:
#             z = net(z)
#         z = self.res3scale(z)
#         for net in self.res3:
#             z = net(z)
#         z = self.res4scale(z)
#         for net in self.res4:
#             z = net(z)

#         z = self.admaxpool(z)

#         conv_out = z+conv_pi
#         conv_out = conv_out.reshape(conv_out.size(0), -1)

#         conv_out = self.fc(conv_out)
#         return conv_out

class ResOnion_PI_uptime(nn.Module):
    def __init__(self, n=100, max_r=100, max_z=100):
        super(ResOnion_PI_uptime, self).__init__()
        self.conv_upsample = ConvEmbModel(max_r, max_z)
        channels = n
        self.res1 = nn.ModuleList([ResidualBasic(channels) for _ in range(2)])
        self.res2scale = Residualscale(channels, 2 * channels)
        self.res2 = nn.ModuleList([ResidualBasic(2 * channels) for _ in range(1)])
        self.res3scale = Residualscale(2 * channels, 4 * channels)
        self.res3 = nn.ModuleList([ResidualBasic(4 * channels) for _ in range(1)])
        self.res4scale = Residualscale(4 * channels, 8 * channels)
        self.res4 = nn.ModuleList([ResidualBasic(8 * channels) for _ in range(1)])
        self.admaxpool = nn.AdaptiveMaxPool2d((3,3))

        final_r = math.ceil(math.ceil(math.ceil(max_r / 2) / 2) / 2)
        final_z = math.ceil(math.ceil(math.ceil(max_z / 2) / 2) / 2)

        self.flatten = nn.Flatten(start_dim=1)

        channels_pi = n
        self.net2 = nn.Sequential(
            nn.Conv2d(in_channels=channels_pi, out_channels=2 * channels_pi, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(num_features=2 * channels_pi),
            nn.ReLU(),
            nn.Conv2d(in_channels=2 * channels_pi, out_channels=2 * channels_pi, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(num_features=2 * channels_pi),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=2 * channels_pi, out_channels=4 * channels_pi, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(num_features=4 * channels_pi),
            nn.ReLU(),
            nn.Conv2d(in_channels=4 * channels_pi, out_channels=4 * channels_pi, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(num_features=4 * channels_pi),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=4 * channels_pi, out_channels=8 * channels_pi, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(num_features=8 * channels_pi),
            nn.ReLU(),
            nn.Conv2d(in_channels=8 * channels_pi, out_channels=8 * channels_pi, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(num_features=8 * channels_pi),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(output_size=(3, 3))
        )
        fc_in_dim = 8 * channels_pi * 3 * 3   # (40,3,3)  (41,3,3)   (81,3,3)
        fc_out_dim = max_r * max_z
        self.fc = nn.Sequential(
            nn.Linear(in_features=fc_in_dim, out_features=fc_out_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=fc_out_dim, out_features=fc_out_dim),
            nn.ReLU()
        )
    def forward(self, input, posi):
        z = self.conv_upsample(input)
        pi = posi

        conv_pi = self.net2(pi)

        for net1 in self.res1:
            z = net1(z)
        z = self.res2scale(z)
        for net2 in self.res2:
            z = net2(z)
        z = self.res3scale(z)
        for net3 in self.res3:
            z = net3(z)
        z = self.res4scale(z)
        for net4 in self.res4:
            z = net4(z.clone())

        z = self.admaxpool(z)

        conv_out = z*conv_pi
        conv_out = conv_out.reshape(conv_out.size(0), -1)

        conv_out = self.fc(conv_out)
        return conv_out

class ResOnion_PI_uptime_softplus(nn.Module):
    def __init__(self, n=100, max_r=100, max_z=100):
        super(ResOnion_PI_uptime_softplus, self).__init__()
        self.conv_upsample = ConvEmbModel(max_r, max_z)
        channels = n
        self.res1 = nn.ModuleList([ResidualBasic(channels) for _ in range(2)])
        self.res2scale = Residualscale(channels, 2 * channels)
        self.res2 = nn.ModuleList([ResidualBasic(2 * channels) for _ in range(1)])
        self.res3scale = Residualscale(2 * channels, 4 * channels)
        self.res3 = nn.ModuleList([ResidualBasic(4 * channels) for _ in range(1)])
        self.res4scale = Residualscale(4 * channels, 8 * channels)
        self.res4 = nn.ModuleList([ResidualBasic(8 * channels) for _ in range(1)])
        self.admaxpool = nn.AdaptiveMaxPool2d((3,3))

        final_r = math.ceil(math.ceil(math.ceil(max_r / 2) / 2) / 2)
        final_z = math.ceil(math.ceil(math.ceil(max_z / 2) / 2) / 2)

        self.flatten = nn.Flatten(start_dim=1)

        channels_pi = n
        self.net2 = nn.Sequential(
            nn.Conv2d(in_channels=channels_pi, out_channels=2 * channels_pi, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(num_features=2 * channels_pi),
            nn.ReLU(),
            nn.Conv2d(in_channels=2 * channels_pi, out_channels=2 * channels_pi, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(num_features=2 * channels_pi),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=2 * channels_pi, out_channels=4 * channels_pi, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(num_features=4 * channels_pi),
            nn.ReLU(),
            nn.Conv2d(in_channels=4 * channels_pi, out_channels=4 * channels_pi, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(num_features=4 * channels_pi),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=4 * channels_pi, out_channels=8 * channels_pi, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(num_features=8 * channels_pi),
            nn.ReLU(),
            nn.Conv2d(in_channels=8 * channels_pi, out_channels=8 * channels_pi, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(num_features=8 * channels_pi),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(output_size=(3, 3))
        )
        fc_in_dim = 8 * channels_pi * 3 * 3   # (40,3,3)  (41,3,3)   (81,3,3)
        fc_out_dim = max_r * max_z
        self.fc = nn.Sequential(
            nn.Linear(in_features=fc_in_dim, out_features=fc_out_dim),
            nn.Softplus(),
            nn.Dropout(0.5),
            nn.Linear(in_features=fc_out_dim, out_features=fc_out_dim),
            nn.Softplus()
        )
    def forward(self, input, posi):
        z = self.conv_upsample(input)
        pi = posi

        conv_pi = self.net2(pi)

        for net in self.res1:
            z = net(z)
        z = self.res2scale(z)
        for net in self.res2:
            z = net(z)
        z = self.res3scale(z)
        for net in self.res3:
            z = net(z)
        z = self.res4scale(z)
        for net in self.res4:
            z = net(z)
        z = self.admaxpool(z)

        conv_out = z*conv_pi
        conv_out = conv_out.reshape(conv_out.size(0), -1)

        conv_out = self.fc(conv_out)
        return conv_out

# class ResOnion_PI_softplus(nn.Module):
#     def __init__(self, n=100, max_r=100, max_z=100):
#         super(ResOnion_PI_softplus, self).__init__()
#         self.conv_upsample = ConvEmbModel(max_r, max_z)
#         channels = n * 2
#         self.res1 = nn.ModuleList([ResidualBasic(channels) for _ in range(2)])
#         self.res2scale = Residualscale(channels, 2 * channels)
#         self.res2 = nn.ModuleList([ResidualBasic(2 * channels) for _ in range(1)])
#         self.res3scale = Residualscale(2 * channels, 4 * channels)
#         self.res3 = nn.ModuleList([ResidualBasic(4 * channels) for _ in range(1)])
#         self.res4scale = Residualscale(4 * channels, 8 * channels)
#         self.res4 = nn.ModuleList([ResidualBasic(8 * channels) for _ in range(1)])

#         final_r = math.ceil(math.ceil(math.ceil(max_r / 2) / 2) / 2)
#         final_z = math.ceil(math.ceil(math.ceil(max_z / 2) / 2) / 2)
#         fc_in_dim = 8 * channels * final_r * final_z
#         fc_out_dim = max_r * max_z
#         self.flatten = nn.Flatten(start_dim=1)
#         self.fc = nn.Sequential(
#             nn.Linear(in_features=fc_in_dim, out_features=fc_out_dim),
#             nn.Softplus(),
#             nn.Dropout(0.5),
#             nn.Linear(in_features=fc_out_dim, out_features=fc_out_dim),
#             nn.Softplus()
#         )

#     def forward(self, input, posi):
#         input = self.conv_upsample(input)
#         z = torch.concat([input, posi], dim=1)
#         for net in self.res1:
#             z = net(z)
#         z = self.res2scale(z)
#         for net in self.res2:
#             z = net(z)
#         z = self.res3scale(z)
#         for net in self.res3:
#             z = net(z)
#         z = self.res4scale(z)
#         for net in self.res4:
#             z = net(z)
#         z = z.reshape(z.size(0), -1)
#         z = self.fc(z)
#         return z

class ResOnion_input(nn.Module):
    def __init__(self, n=100, max_r=100, max_z=100):
        super(ResOnion_input, self).__init__()
        self.conv_upsample = ConvEmbModel(max_r, max_z)
        channels = n
        self.res1 = nn.ModuleList([ResidualBasic(channels) for _ in range(2)])
        self.res2scale = Residualscale(channels, 2 * channels)
        self.res2 = nn.ModuleList([ResidualBasic(2 * channels) for _ in range(1)])
        self.res3scale = Residualscale(2 * channels, 4 * channels)
        self.res3 = nn.ModuleList([ResidualBasic(4 * channels) for _ in range(1)])
        self.res4scale = Residualscale(4 * channels, 8 * channels)
        self.res4 = nn.ModuleList([ResidualBasic(8 * channels) for _ in range(1)])
        self.admaxpool = nn.AdaptiveMaxPool2d((3,3))

        # final_r = math.ceil(math.ceil(math.ceil(max_r / 2) / 2) / 2)
        # final_z = math.ceil(math.ceil(math.ceil(max_z / 2) / 2) / 2)
        fc_in_dim = 8 * channels * 3 * 3
        fc_out_dim = max_r * max_z

        self.fc = nn.Sequential(
            nn.Linear(in_features=fc_in_dim, out_features=fc_out_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=fc_out_dim, out_features=fc_out_dim),
            nn.ReLU()
        )

    def forward(self, input):
        input = self.conv_upsample(input)
        for net in self.res1:
            z = net(input)
        z = self.res2scale(z)
        for net in self.res2:
            z = net(z)
        z = self.res3scale(z)
        for net in self.res3:
            z = net(z)
        z = self.res4scale(z)
        for net in self.res4:
            z = net(z)
            
        z = self.admaxpool(z)
        
        z = z.reshape(z.size(0), -1)
        z = self.fc(z)
        return z
    
class ResOnion_input_softplus(nn.Module):
    def __init__(self, n=100, max_r=100, max_z=100):
        super(ResOnion_input_softplus, self).__init__()
        self.conv_upsample = ConvEmbModel(max_r, max_z)
        channels = n
        self.res1 = nn.ModuleList([ResidualBasic(channels) for _ in range(2)])
        self.res2scale = Residualscale(channels, 2 * channels)
        self.res2 = nn.ModuleList([ResidualBasic(2 * channels) for _ in range(1)])
        self.res3scale = Residualscale(2 * channels, 4 * channels)
        self.res3 = nn.ModuleList([ResidualBasic(4 * channels) for _ in range(1)])
        self.res4scale = Residualscale(4 * channels, 8 * channels)
        self.res4 = nn.ModuleList([ResidualBasic(8 * channels) for _ in range(1)])
        self.admaxpool = nn.AdaptiveMaxPool2d((3,3))

        # final_r = math.ceil(math.ceil(math.ceil(max_r / 2) / 2) / 2)
        # final_z = math.ceil(math.ceil(math.ceil(max_z / 2) / 2) / 2)
        fc_in_dim = 8 * channels * 3 * 3
        fc_out_dim = max_r * max_z

        self.fc = nn.Sequential(
            nn.Linear(in_features=fc_in_dim, out_features=fc_out_dim),
            nn.Softplus(),
            nn.Dropout(0.5),
            nn.Linear(in_features=fc_out_dim, out_features=fc_out_dim),
            nn.Softplus()
        )

    def forward(self, input):
        input = self.conv_upsample(input)
        for net in self.res1:
            z = net(input)
        z = self.res2scale(z)
        for net in self.res2:
            z = net(z)
        z = self.res3scale(z)
        for net in self.res3:
            z = net(z)
        z = self.res4scale(z)
        for net in self.res4:
            z = net(z)

        z = self.admaxpool(z)
        
        z = z.reshape(z.size(0), -1)
        z = self.fc(z)
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

if __name__ =="__main__":
    n = 40
    r = 32
    z = 36
    flatten_len = r*z
    input_shapes = [(1,n), (1,n,z,r)]
    # input_shapes = [(1,n)]
    # inputs = [torch.randn(*shape) for shape in input_shapes]
    inputs = [torch.randn(*shape) for shape in input_shapes]
    # onion = Onion_input(n=n, max_r=r, max_z=z)
    onion = ResOnion_PI_up(n=n, max_r=r, max_z=z)

    # 将summary输出保存到文本文件中
    summary(onion, input_data=inputs)
    with open('onion_input_model.txt', 'w') as f:
        with redirect_stdout(f):
            summary(onion, input_data=inputs)