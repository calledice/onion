from dataset import OnionDataset
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
import torch.nn.functional as F
import math
class Config:
    def __init__(self, n_layer, n_head, dropout, bias, dtype, batch_size,max_input_len,max_rz_len):
        self.n_layer = n_layer
        self.n_head = n_head
        self.dropout = dropout
        self.bias = bias
        self.dtype = dtype
        self.batch_size = batch_size
        self.max_input_len = max_input_len
        self.max_rz_len = max_rz_len
class ConvEmbModel(nn.Module):
    def __init__(self, max_r=100, max_z=100, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.r = max_r
        self.z = max_z
        out_channels = max_r * max_z
        self.device = device  # Explicitly initialize device
        # self.sigmoid = nn.Sigmoid()  # Corrected typo in the activation function name
        self.conv = nn.Conv1d(in_channels=1, out_channels=out_channels, kernel_size=1)
        # self.batch_norm = nn.BatchNorm1d(input_len)

    def forward(self, x):
        x = self.conv(x.unsqueeze(1)).transpose(1, 2)
        x = x.reshape(x.shape[0], x.shape[1], self.r, self.z)
        return x

class Onion(nn.Module):
    def __init__(self, n=100, max_r=100, max_z=100, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super(Onion, self).__init__()
        self.conv_upsample = ConvEmbModel(max_r, max_z, device)
        channels = max_r + max_z + 1

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
            nn.Conv2d(in_channels=channels, out_channels=channels*2, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channels*2, out_channels=channels*2, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(num_features=channels*2),
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


dataset = OnionDataset("../data_Phantom/phantomdata/mini_1_test_database.h5")
train_loader = DataLoader(dataset, batch_size=4, shuffle=False)
onion = Onion()
loss_fn = nn.MSELoss()
optim = torch.optim.Adam(params=onion.parameters(), lr=0.01)
for (input, regi, posi, info), label in train_loader:
    pred = onion(input, regi, posi)
    optim.zero_grad()
    loss = loss_fn(pred, label)
    loss.backward()
    optim.step()
    print(loss)

