from dataset import OnionDataset
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
import torch.nn.functional as F
import math


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
    def __init__(self, input_len=100, out_channels=10000, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.device = device  # Explicitly initialize device
        self.conv = nn.Conv1d(in_channels=1, out_channels=out_channels, kernel_size=1)
        # self.batch_norm = nn.BatchNorm1d(input_len)

    def forward(self, x):
        x = self.conv(x.unsqueeze(1)).transpose(1, 2)
        return x


class Onion(nn.Module):
    def __init__(self, config):
        super(Onion, self).__init__()
        self.config = config
        self.conv_emb = ConvEmbModel(out_channels=config.max_rz_len)

    def _init_weihts(self):
        # 实现权重初始化逻辑
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)  # 举例使用Xavier初始化，根据需要调整
            elif 'bias' in name and param is not None:
                nn.init.zeros_(param)

    def forward(self, input, regi, posi):
        embeded_input = self.conv_emb(input)
        posi = self.scale_linear(posi)
        # vec = embeded_input + regi + posi
        vec = (embeded_input + posi) * regi
        for block in self.block_stack:
            vec = block(vec)
        x_in = torch.sum(vec, dim=1, dtype=torch.float32)  # todo 直接加不合适
        # x_in = self.LN(x_in)
        output = self.out_head(x_in).unsqueeze(1)
        return output

# max_input_len, max_rz_len = 100, 2500
# train_path = '/media/congwang/data/python_code/Onion/data_Phantom/phantomdata/mini_train_database.h5'
# train_dataset = OnionDataset(train_path, max_input_len=max_input_len, max_rz_len=max_rz_len)
# train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False)
#
# loss_fn = nn.MSELoss().to('cuda')
# conv_emb = ConvEmbModel(out_channels=max_rz_len)
# config = Config(1024, 40, 6, 10, max_rz_len, 1, 0.0, True, torch.float32, 3, 1, 1)
# model = Onion(config)
# for (input, regi, posi, info), label in train_loader:
#     input = conv_emb(input)
#     regi = regi.unsqueeze(1).repeat(1, max_input_len, 1)
#     vec = input + regi + posi
#     pred = model(vec).squeeze(1)
#     loss = loss_fn(pred, label)

