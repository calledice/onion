from dataset import OnionDataset
from torch.utils.data import Dataset, DataLoader
from model_without_regi import *
import torch.nn as nn
import torch
from tqdm import tqdm
import os
import json

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


train_set = OnionDataset("../data_Phantom/phantomdata/mini_2_train_database.h5")
val_set = OnionDataset("../data_Phantom/phantomdata/mini_2_valid_database.h5")

# 临时加的，为了不做padding
n = int(train_set.input_len_org[0])
r = int(train_set.info_list[0][1])
z = int(train_set.info_list[0][2])

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64, shuffle=True)
onion = Onion(n=n, max_r=r, max_z=z)
onion.to(device)

loss_fn = nn.MSELoss()
optim = torch.optim.Adam(params=onion.parameters(), lr=0.001)
weight = torch.tensor([3.0], requires_grad=True, dtype=torch.float32).to(device)
out_dir = 'output/train'
os.makedirs(out_dir, exist_ok=True)
min_val_loss = float('inf')
early_stop = 5

train_losses = []
val_losses = []
epochs = 100
for epoch in range(epochs):
    # 训练阶段
    onion.train()
    print(f"epoch: {epoch}")
    losses = []
    for (input, regi, posi, info), label in tqdm(train_loader, desc="Training"):
        input, regi, posi, label = input.to(device), regi.to(device), posi.to(device), label.to(device)
        pred = onion(input)
        optim.zero_grad()
        loss = weighted_mse_loss(pred, label, 10)
        loss.backward()
        optim.step()
        losses.append(loss.item())
    train_loss = sum(losses) / len(losses)
    print(f"epoch{epoch} training loss: {train_loss}")
    json.dump(losses, open(f"{out_dir}/training_loss{epoch}.json", 'w'))
    train_losses.append(train_loss)

    # 验证阶段
    onion.eval()
    best_epoch = -1
    losses = []
    preds = []
    labels = []
    for (input, regi, posi, info), label in tqdm(val_loader, desc="Validating"):
        input, regi, posi, label = input.to(device), regi.to(device), posi.to(device), label.to(device)
        pred = onion(input)
        loss = weighted_mse_loss(pred, label, 10)
        preds.append(pred.reshape(-1, r, z))
        labels.append(label.reshape(-1, r, z))
        losses.append(loss.item())
    val_loss = sum(losses) / len(losses)
    print(f"epoch{epoch} validation loss: {val_loss}")
    print(f"epoch{epoch} min loss: {min_val_loss}")
    if val_loss < min_val_loss:
        min_val_loss = val_loss
        torch.save(onion, f"{out_dir}/model_best.pth")
        with open(f"{out_dir}/best_epoch.txt", 'w') as f:
            f.write(str(epoch))
        early_stop = 5
    json.dump(losses, open(f"{out_dir}/val_loss{epoch}.json", 'w'))
    val_losses.append(val_loss)
    early_stop -= 1
    if early_stop <= 0:
        break

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

plt.savefig('loss_curve.png')
# 显示图形
plt.show()
