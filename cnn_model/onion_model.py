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
            n_head=None, dropout=None, bias=True, dtype=torch.float32, batch_size=64,
            max_n=100, max_r=100, max_z=100, lr=0.001, epochs=20, early_stop=5,lambda_l1 = 0.01,p=2 ):
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
        self.lambda_l1 = lambda_l1
        self.randomnumseed = randomnumseed
        self.p = p

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

class ResidualBasic(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU()
        )

    def forward(self, x):
        return self.features(x) + x

class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
            nn.ReLU()
        )

    def forward(self, x):
        return self.feature(x)

class Onion_gavin(nn.Module):
    def __init__(self, n=100, max_r=100, max_z=100):
        super(Onion_gavin, self).__init__()
        self.conv_upsample = ConvEmbModel(max_r, max_z)
        channels = n * 2 + 1

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.GELU(),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(3, 3), padding=(1, 1)),
            # nn.BatchNorm2d(num_features=channels),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.GELU(),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(3, 3), padding=(1, 1)),
            # nn.BatchNorm2d(num_features=channels),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=channels, out_channels=channels * 2, kernel_size=(3, 3), padding=(1, 1)),
            nn.GELU(),
            nn.Conv2d(in_channels=channels * 2, out_channels=channels * 2, kernel_size=(3, 3), padding=(1, 1)),
            # nn.BatchNorm2d(num_features=channels * 2),
            nn.GELU(),
            nn.AdaptiveMaxPool2d(output_size=(10, 10))
        )

        conv_out_dim = channels * 2 * 10 * 10
        fc_out_dim = max_r * max_z

        self.fc = nn.Sequential(
            nn.Linear(in_features=conv_out_dim, out_features=fc_out_dim),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=fc_out_dim, out_features=fc_out_dim),
            nn.ReLU()
        )

    def forward(self, input, regi, posi):
        input = self.conv_upsample(input)
        regi = regi.unsqueeze(dim=0).transpose(0, 1)
        final_input = torch.concat([input, regi, posi], dim=1)
        conv_out = self.net(final_input)    
        conv_out = conv_out.reshape(conv_out.size(0), -1)
        out = self.fc(conv_out)
        return out

class Onion_gavin_original(nn.Module):
    def __init__(self, n=100, max_r=100, max_z=100):
        super(Onion_gavin_original, self).__init__()
        self.conv_upsample = ConvEmbModel(max_r, max_z)
        channels = n * 2 + 1

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.GELU(),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(num_features=channels),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.GELU(),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(num_features=channels),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=channels, out_channels=channels * 2, kernel_size=(3, 3), padding=(1, 1)),
            nn.GELU(),
            nn.Conv2d(in_channels=channels * 2, out_channels=channels * 2, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(num_features=channels * 2),
            nn.GELU(),
            nn.AdaptiveMaxPool2d(output_size=(10, 10))
        )

        conv_out_dim = channels * 2 * 10 * 10
        fc_out_dim = max_r * max_z

        self.fc = nn.Sequential(
            nn.Linear(in_features=conv_out_dim, out_features=fc_out_dim),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=fc_out_dim, out_features=fc_out_dim),
            nn.ReLU()
        )

    def forward(self, input, regi, posi):
        input = self.conv_upsample(input)
        regi = regi.unsqueeze(dim=0).transpose(0, 1)
        final_input = torch.concat([input, regi, posi], dim=1)
        conv_out = self.net(final_input)    
        conv_out = conv_out.reshape(conv_out.size(0), -1)
        out = self.fc(conv_out)
        return out

class Onion_PI(nn.Module):
    def __init__(self, n=100, max_r=100, max_z=100):
        super(Onion_PI, self).__init__()
        self.conv_upsample = ConvEmbModel(max_r, max_z)
        channels = n * 2 + 1

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=2*channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=2*channels, out_channels=2*channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(num_features=2*channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(in_channels=2*channels, out_channels=4*channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=4*channels, out_channels=4*channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=4*channels, out_channels=4*channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(num_features=4*channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(in_channels=4*channels, out_channels=8*channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=8*channels, out_channels=8*channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=8*channels, out_channels=8*channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(num_features=8*channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d(output_size=(2, 2))
        )

        conv_out_dim = 8*channels*2*2
        fc_out_dim = max_r * max_z

        self.fc = nn.Sequential(
            nn.Linear(in_features=conv_out_dim, out_features=fc_out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(in_features=fc_out_dim, out_features=fc_out_dim),
            nn.ReLU()
            # nn.Softplus()
        )

    def forward(self, input, regi, posi):
        input = self.conv_upsample(input)
        regi = regi.unsqueeze(dim=0).transpose(0, 1)
        # 在 posi 上加一个非常小的正值
        # posi = posi + 1e-6
        final_input = torch.concat([input, regi, posi], dim=1)
        conv_out = self.net(final_input)
        # conv_out = self.deconv_net(conv_out)
        conv_out = conv_out.reshape(conv_out.size(0), -1)
        out = self.fc(conv_out)
        return out
class Onion_PI_softplus(nn.Module):
    def __init__(self, n=100, max_r=100, max_z=100):
        super(Onion_PI, self).__init__()
        self.conv_upsample = ConvEmbModel(max_r, max_z)
        channels = n * 2 + 1

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=2*channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=2*channels, out_channels=2*channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(num_features=2*channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(in_channels=2*channels, out_channels=4*channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=4*channels, out_channels=4*channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=4*channels, out_channels=4*channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(num_features=4*channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(in_channels=4*channels, out_channels=8*channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=8*channels, out_channels=8*channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=8*channels, out_channels=8*channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(num_features=8*channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d(output_size=(2, 2))
        )

        conv_out_dim = 8*channels*2*2
        fc_out_dim = max_r * max_z

        self.fc = nn.Sequential(
            nn.Linear(in_features=conv_out_dim, out_features=fc_out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(in_features=fc_out_dim, out_features=fc_out_dim),
            # nn.ReLU()
            nn.Softplus()
        )

    def forward(self, input, regi, posi):
        input = self.conv_upsample(input)
        regi = regi.unsqueeze(dim=0).transpose(0, 1)
        # 在 posi 上加一个非常小的正值
        # posi = posi + 1e-6
        final_input = torch.concat([input, regi, posi], dim=1)
        conv_out = self.net(final_input)
        # conv_out = self.deconv_net(conv_out)
        conv_out = conv_out.reshape(conv_out.size(0), -1)
        out = self.fc(conv_out)
        return out

class Onion_PI_posiplus(nn.Module):
    def __init__(self, n=100, max_r=100, max_z=100):
        super(Onion_PI, self).__init__()
        self.conv_upsample = ConvEmbModel(max_r, max_z)
        channels = n * 2 + 1

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=2*channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=2*channels, out_channels=2*channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(num_features=2*channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(in_channels=2*channels, out_channels=4*channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=4*channels, out_channels=4*channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=4*channels, out_channels=4*channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(num_features=4*channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(in_channels=4*channels, out_channels=8*channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=8*channels, out_channels=8*channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=8*channels, out_channels=8*channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(num_features=8*channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d(output_size=(2, 2))
        )

        conv_out_dim = 8*channels*2*2
        fc_out_dim = max_r * max_z

        self.fc = nn.Sequential(
            nn.Linear(in_features=conv_out_dim, out_features=fc_out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(in_features=fc_out_dim, out_features=fc_out_dim),
            nn.ReLU()
            # nn.Softplus()
        )

    def forward(self, input, regi, posi):
        input = self.conv_upsample(input)
        regi = regi.unsqueeze(dim=0).transpose(0, 1)
        # 在 posi 上加一个非常小的正值
        posi = posi + 1e-6
        final_input = torch.concat([input, regi, posi], dim=1)
        conv_out = self.net(final_input)
        # conv_out = self.deconv_net(conv_out)
        conv_out = conv_out.reshape(conv_out.size(0), -1)
        out = self.fc(conv_out)
        return out

class Onion_PI_softplus_posiplus(nn.Module):
    def __init__(self, n=100, max_r=100, max_z=100):
        super(Onion_PI, self).__init__()
        self.conv_upsample = ConvEmbModel(max_r, max_z)
        channels = n * 2 + 1

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=2*channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=2*channels, out_channels=2*channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(num_features=2*channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(in_channels=2*channels, out_channels=4*channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=4*channels, out_channels=4*channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=4*channels, out_channels=4*channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(num_features=4*channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(in_channels=4*channels, out_channels=8*channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=8*channels, out_channels=8*channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=8*channels, out_channels=8*channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(num_features=8*channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d(output_size=(2, 2))
        )

        conv_out_dim = 8*channels*2*2
        fc_out_dim = max_r * max_z

        self.fc = nn.Sequential(
            nn.Linear(in_features=conv_out_dim, out_features=fc_out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(in_features=fc_out_dim, out_features=fc_out_dim),
            # nn.ReLU()
            nn.Softplus()
        )

    def forward(self, input, regi, posi):
        input = self.conv_upsample(input)
        regi = regi.unsqueeze(dim=0).transpose(0, 1)
        # 在 posi 上加一个非常小的正值
        posi = posi + 1e-6
        final_input = torch.concat([input, regi, posi], dim=1)
        conv_out = self.net(final_input)
        # conv_out = self.deconv_net(conv_out)
        conv_out = conv_out.reshape(conv_out.size(0), -1)
        out = self.fc(conv_out)
        return out

class Onion_input(nn.Module):
    def __init__(self, n=100, max_r=100, max_z=100):
        super(Onion_input, self).__init__()
        self.conv_upsample = ConvEmbModel(max_r, max_z)
        # channels = n * 2 + 1
        channels = n
        # channels = 64

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=2*channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=2*channels, out_channels=2*channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(num_features=2*channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(in_channels=2*channels, out_channels=4*channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=4*channels, out_channels=4*channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=4*channels, out_channels=4*channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(num_features=4*channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(in_channels=4*channels, out_channels=8*channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=8*channels, out_channels=8*channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=8*channels, out_channels=8*channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(num_features=8*channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d(output_size=(2, 2))
        )

        conv_out_dim = 8*channels*2*2
        fc_out_dim = max_r * max_z

        self.fc = nn.Sequential(
            nn.Linear(in_features=conv_out_dim, out_features=fc_out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(in_features=fc_out_dim, out_features=fc_out_dim),
            nn.ReLU()
            # nn.Softplus()
        )

    def forward(self, input):
        input = self.conv_upsample(input)
        conv_out = self.net(input)
        conv_out = conv_out.reshape(conv_out.size(0), -1)
        out = self.fc(conv_out)
        return out
class Onion_input_softplus(nn.Module):
    def __init__(self, n=100, max_r=100, max_z=100):
        super(Onion_input, self).__init__()
        self.conv_upsample = ConvEmbModel(max_r, max_z)
        # channels = n * 2 + 1
        channels = n
        # channels = 64

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=2*channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=2*channels, out_channels=2*channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(num_features=2*channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(in_channels=2*channels, out_channels=4*channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=4*channels, out_channels=4*channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=4*channels, out_channels=4*channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(num_features=4*channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(in_channels=4*channels, out_channels=8*channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=8*channels, out_channels=8*channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=8*channels, out_channels=8*channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(num_features=8*channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d(output_size=(2, 2))
        )

        conv_out_dim = 8*channels*2*2
        fc_out_dim = max_r * max_z

        self.fc = nn.Sequential(
            nn.Linear(in_features=conv_out_dim, out_features=fc_out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(in_features=fc_out_dim, out_features=fc_out_dim),
            # nn.ReLU()
            nn.Softplus()
        )

    def forward(self, input):
        input = self.conv_upsample(input)
        conv_out = self.net(input)
        conv_out = conv_out.reshape(conv_out.size(0), -1)
        out = self.fc(conv_out)
        return out
    
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
            nn.ReLU()
        )

    def forward(self, x):
        z = self.fc1(x)
        z = z.reshape(-1, self.n, self.r, self.z)
        z = self.conv(z)
        z = z.reshape(-1, self.out_dim)
        z = self.fc2(z)
        return z

class ResOnion_PI(nn.Module):
    def __init__(self, n=100, max_r=100, max_z=100):
        super(ResOnion_PI, self).__init__()
        self.conv_upsample = ConvEmbModel(max_r, max_z)
        channels = n * 2 + 1
        self.res1 = nn.ModuleList([ResidualBasic(channels) for _ in range(4)])
        self.down1 = DownSample(channels, 2 * channels)
        self.res2 = nn.ModuleList([ResidualBasic(2 * channels) for _ in range(8)])
        self.down2 = DownSample(2 * channels, 4 * channels)
        self.res3 = nn.ModuleList([ResidualBasic(4 * channels) for _ in range(12)])

        final_r = math.ceil(math.ceil(max_r / 2) / 2)
        final_z = math.ceil(math.ceil(max_z / 2) / 2)
        fc_in_dim = 4 * channels * final_r * final_z
        fc_out_dim = max_r * max_z
        self.flatten = nn.Flatten(start_dim=1)
        self.fc = nn.Sequential(
            nn.Linear(in_features=fc_in_dim, out_features=fc_out_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=fc_out_dim, out_features=fc_out_dim),
            nn.ReLU()
        )

    def forward(self, input, regi, posi):
        input = self.conv_upsample(input)
        regi = regi.unsqueeze(dim=0).transpose(0, 1)
        z = torch.concat([input, regi, posi], dim=1)
        for net in self.res1:
            z = net(z)
        z = self.down1(z)
        for net in self.res2:
            z = net(z)
        z = self.down2(z)
        for net in self.res3:
            z = net(z)
        z = z.reshape(z.size(0), -1)
        z = self.fc(z)
        return z
class ResOnion_PI_softplus(nn.Module):
    def __init__(self, n=100, max_r=100, max_z=100):
        super(ResOnion_PI, self).__init__()
        self.conv_upsample = ConvEmbModel(max_r, max_z)
        channels = n * 2 + 1
        self.res1 = nn.ModuleList([ResidualBasic(channels) for _ in range(4)])
        self.down1 = DownSample(channels, 2 * channels)
        self.res2 = nn.ModuleList([ResidualBasic(2 * channels) for _ in range(8)])
        self.down2 = DownSample(2 * channels, 4 * channels)
        self.res3 = nn.ModuleList([ResidualBasic(4 * channels) for _ in range(12)])

        final_r = math.ceil(math.ceil(max_r / 2) / 2)
        final_z = math.ceil(math.ceil(max_z / 2) / 2)
        fc_in_dim = 4 * channels * final_r * final_z
        fc_out_dim = max_r * max_z
        self.flatten = nn.Flatten(start_dim=1)
        self.fc = nn.Sequential(
            nn.Linear(in_features=fc_in_dim, out_features=fc_out_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=fc_out_dim, out_features=fc_out_dim),
            nn.Softplus()
        )

    def forward(self, input, regi, posi):
        input = self.conv_upsample(input)
        regi = regi.unsqueeze(dim=0).transpose(0, 1)
        z = torch.concat([input, regi, posi], dim=1)
        for net in self.res1:
            z = net(z)
        z = self.down1(z)
        for net in self.res2:
            z = net(z)
        z = self.down2(z)
        for net in self.res3:
            z = net(z)
        z = z.reshape(z.size(0), -1)
        z = self.fc(z)
        return z

class ResOnion_input(nn.Module):
    def __init__(self, n=100, max_r=100, max_z=100):
        super(ResOnion_input, self).__init__()
        self.conv_upsample = ConvEmbModel(max_r, max_z)
        channels = n
        self.res1 = nn.ModuleList([ResidualBasic(channels) for _ in range(4)])
        self.down1 = DownSample(channels, 2 * channels)
        self.res2 = nn.ModuleList([ResidualBasic(2 * channels) for _ in range(8)])
        self.down2 = DownSample(2 * channels, 4 * channels)
        self.res3 = nn.ModuleList([ResidualBasic(4 * channels) for _ in range(12)])

        final_r = math.ceil(math.ceil(max_r / 2) / 2)
        final_z = math.ceil(math.ceil(max_z / 2) / 2)
        fc_in_dim = 4 * channels * final_r * final_z
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
        z = self.down1(z)
        for net in self.res2:
            z = net(z)
        z = self.down2(z)
        for net in self.res3:
            z = net(z)
        z = z.reshape(z.size(0), -1)
        z = self.fc(z)
        return z
class ResOnion_input_softplus(nn.Module):
    def __init__(self, n=100, max_r=100, max_z=100):
        super(ResOnion_input, self).__init__()
        self.conv_upsample = ConvEmbModel(max_r, max_z)
        channels = n
        self.res1 = nn.ModuleList([ResidualBasic(channels) for _ in range(4)])
        self.down1 = DownSample(channels, 2 * channels)
        self.res2 = nn.ModuleList([ResidualBasic(2 * channels) for _ in range(8)])
        self.down2 = DownSample(2 * channels, 4 * channels)
        self.res3 = nn.ModuleList([ResidualBasic(4 * channels) for _ in range(12)])

        final_r = math.ceil(math.ceil(max_r / 2) / 2)
        final_z = math.ceil(math.ceil(max_z / 2) / 2)
        fc_in_dim = 4 * channels * final_r * final_z
        fc_out_dim = max_r * max_z

        self.fc = nn.Sequential(
            nn.Linear(in_features=fc_in_dim, out_features=fc_out_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=fc_out_dim, out_features=fc_out_dim),
            nn.Softplus()
        )

    def forward(self, input):
        input = self.conv_upsample(input)
        for net in self.res1:
            z = net(input)
        z = self.down1(z)
        for net in self.res2:
            z = net(z)
        z = self.down2(z)
        for net in self.res3:
            z = net(z)
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
    # input_shapes = [(1,n),(1,r,z),(1,n,r,z)]
    input_shapes = [(1,n)]
    inputs = [torch.randn(*shape) for shape in input_shapes]
    onion = Onion_input(n=n, max_r=r, max_z=z)

    # 将summary输出保存到文本文件中
    summary(onion, input_data=inputs)
    with open('onion_input_model.txt', 'w') as f:
        with redirect_stdout(f):
            summary(onion, input_data=inputs)