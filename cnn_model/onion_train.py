from dataset import OnionDataset
from torch.utils.data import Dataset, DataLoader
from onion_model import *
import torch.nn as nn
import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
import math
import json
from onion_model import Onion



os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n = 34
r = 21
z = 31

dataset = OnionDataset("../data_Phantom/phantomdata/mini_2_train_database.h5", max_input_len=n, max_r=r, max_z=z)
print(len(dataset))
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
onion = Onion(n=n, max_r=r, max_z=z)
onion.to(device)
onion.train()
loss_fn = nn.MSELoss()
optim = torch.optim.Adam(params=onion.parameters(), lr=0.001)
weight = torch.tensor([3.0], requires_grad=True, dtype=torch.float32).to(device)
out_dir = 'output/train'
os.makedirs(out_dir, exist_ok=True)

epochs = 100
for epoch in range(epochs):
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
    print(sum(losses) / len(losses))
    json.dump(losses, open(f"{out_dir}/training_loss{epoch}.json", 'w'))
    torch.save(onion, f"{out_dir}/model{epoch}.pth")
