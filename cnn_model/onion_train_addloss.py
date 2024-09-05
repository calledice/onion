from dataset import OnionDataset
from torch.utils.data import Dataset, DataLoader
from model_without_regi import *
# from onion_model import *
import torch.nn as nn
import torch
from tqdm import tqdm
import os
import json


class Config:
    def __init__(self, n_layer=None, n_head=None, dropout=None, bias=True, dtype=torch.float32, batch_size=64, 
                 max_input_len=100, max_rz_len=10000, max_n=100, max_r=100, max_z=100, lr=0.001, epochs=20, early_stop=5):
        self.n_layer = n_layer
        self.n_head = n_head
        self.dropout = dropout
        self.bias = bias
        self.dtype = dtype
        self.batch_size = batch_size
        self.max_input_len = max_input_len
        self.max_rz_len = max_rz_len
        self.max_n = max_n
        self.max_r = max_r
        self.max_z = max_z
        self.lr = lr
        self.epochs = epochs
        self.early_stop = early_stop
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, train_loader, val_loader, out_dir, config:Config):
    epochs = config.epochs
    early_stop = config.early_stop
    device = config.device
    os.makedirs(f'{out_dir}/train', exist_ok=True)
    min_val_loss = float('inf')
    optim = torch.optim.Adam(params=model.parameters(), lr=config.lr)
    train_losses = []
    val_losses = []
    loss_mse = nn.MSELoss()
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        print(f"epoch: {epoch}")
        losses = []
        for (input, regi, posi, info), label in tqdm(train_loader, desc="Training"):
            input, regi, posi, label = input.to(device), regi.to(device), posi.to(device), label.to(device)
            pred = model(input)
            # pred = model(input, regi, posi)
            pred_temp = pred.unsqueeze(-1)
            result = torch.bmm(posi.view(len(posi), len(posi[0]), -1), pred_temp).squeeze(-1)

            optim.zero_grad()
            loss = weighted_mse_loss(pred, label, 10)+ 10.0*loss_mse(input,result)
            loss.backward()
            optim.step()
            losses.append(loss.item())
        train_loss = sum(losses) / len(losses)
        print(f"epoch{epoch} training loss: {train_loss}")
        json.dump(losses, open(f"{out_dir}/train/training_loss{epoch}.json", 'w'))
        train_losses.append(train_loss)

        # 验证阶段
        model.eval()
        best_epoch = -1
        losses = []
        preds = []
        labels = []
        for (input, regi, posi, info), label in tqdm(val_loader, desc="Validating"):
            input, regi, posi, label = input.to(device), regi.to(device), posi.to(device), label.to(device)
            pred = model(input)
            # pred = model(input, regi, posi)
            loss = weighted_mse_loss(pred, label, 10)
            preds.append(pred.detach().reshape(-1, config.max_r, config.max_z))
            labels.append(label.detach().reshape(-1, config.max_r, config.max_z))
            losses.append(loss.detach().item())
        val_loss = sum(losses) / len(losses)
        print(f"epoch{epoch} validation loss: {val_loss}")
        print(f"epoch{epoch} min loss: {min_val_loss}")
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save(model, f"{out_dir}/train/model_best.pth")
            with open(f"{out_dir}/train/best_epoch.txt", 'w') as f:
                f.write(str(epoch))
            if early_stop > 0:
                early_stop = 5
        json.dump(losses, open(f"{out_dir}/train/val_loss{epoch}.json", 'w'))
        val_losses.append(val_loss)
        if early_stop >= 0:
            early_stop -= 1
            if early_stop <= 0:
                break
    return train_losses, val_losses

def plot_loss(train_losses, val_losses, out_dir):

    import matplotlib.pyplot as plt
    iters = list(range(len(train_losses)))

    # 创建一个新的图形
    plt.figure()

    # 绘制第一条曲线
    plt.plot(iters, train_losses, label='training loss', color='blue')

    # 绘制第二条曲线
    plt.plot(iters, val_losses, label='validation loss', color='red')

    # 添加标题和标签
    plt.title('Loss curve')
    plt.xlabel('iters')
    plt.ylabel('loss')

    # 显示图例
    plt.legend()

    plt.savefig(f'{out_dir}/train/loss_curve.png')
    # 显示图形
    plt.show()

def run(train_path, val_path, test_path, out_dir, config):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = config.device

    train_set = OnionDataset(train_path)
    val_set = OnionDataset(val_path)

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=True)

    # 临时加的，为了不做padding
    n = int(train_set.input_len_org[0])
    r = int(train_set.info_list[0][1])
    z = int(train_set.info_list[0][2])
    config.max_n = n
    config.max_r = r
    config.max_z = z
    onion = Onion(n=n, max_r=r, max_z=z)
    onion.to(device)

    train_losses, val_losses = train(onion, train_loader, val_loader, out_dir, config=config)
    plot_loss(train_losses, val_losses, out_dir)

if __name__ == '__main__':
    train_path = "../data_Phantom/phantomdata/mini_1_train_database_1_100_1000.h5"
    val_path = "../data_Phantom/phantomdata/mini_1_valid_database_1_100_1000.h5"
    test_path = "../data_Phantom/phantomdata/mini_1_test_database_1_100_1000.h5"
    out_dir = "output/Phantom_addloss"
    config = Config(early_stop=-1, epochs=18,batch_size=64)
    run(train_path, val_path, test_path, out_dir, config)