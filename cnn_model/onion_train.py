from dataset import OnionDataset
from torch.utils.data import Dataset, DataLoader
from onion_model import *
import torch.nn as nn
import torch
from tqdm import tqdm
import os
import json
from onion_model import Onion

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n = 34
r = 21
z = 31

train_set = OnionDataset("../data_Phantom/phantomdata/mini_2_train_database.h5", max_input_len=n, max_r=r, max_z=z)
val_set = OnionDataset("../data_Phantom/phantomdata/mini_2_valid_database.h5", max_input_len=n, max_r=r, max_z=z)
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

epochs = 100
for epoch in range(epochs):
    # 训练阶段
    onion.train()
    print(f"epoch: {epoch}")
    losses = []
    for (input, regi, posi, info), label in tqdm(train_loader, desc="Training"):
        input, regi, posi, label = input.to(device), regi.to(device), posi.to(device), label.to(device)
        pred = onion(input, regi, posi)
        optim.zero_grad()
        loss = weighted_mse_loss(pred, label, 10)
        loss.backward()
        optim.step()
        losses.append(loss.item())
    print(f"epoch{epoch} training loss: {sum(losses) / len(losses)}")
    json.dump(losses, open(f"{out_dir}/training_loss{epoch}.json", 'w'))

    # 验证阶段
    onion.eval()
    best_epoch = -1
    losses = []
    preds = []
    labels = []
    for (input, regi, posi, info), label in tqdm(val_loader, desc="Validating"):
        input, regi, posi, label = input.to(device), regi.to(device), posi.to(device), label.to(device)
        pred = onion(input, regi, posi)
        loss = weighted_mse_loss(pred, label, 10)
        loss.backward()
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
    json.dump(losses, open(f"{out_dir}/val_loss{epoch}.json", 'w'))
