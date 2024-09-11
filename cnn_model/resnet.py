import torch 
import torch.nn as nn
import math


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


class ResOnion(nn.Module):
    def __init__(self, n=100, max_r=100, max_z=100):
        super(ResOnion, self).__init__()
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
        return z

if __name__ == '__main__':
    basic = ResOnion(40, 34, 27)

    input = torch.rand(64, 40)
    posi = torch.rand(64, 40, 34, 27)
    regi = torch.rand(64, 34, 27)
    x = basic(input, regi, posi)
    print(x.shape)
    # print(basic)