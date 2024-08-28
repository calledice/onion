import torch 
print(torch.cuda.is_available())
from dataset import OnionDataset
from onion_model import Onion, ConvEmbModel, weighted_mse_loss
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json
import os
n = 34
r = 21
z = 31
print(torch.cuda.is_available())

dataset = OnionDataset("./data_Phantom/phantomdata/mini_2_test_database.h5", max_input_len=n, max_r=r, max_z=z)
print(len(dataset))
batch_size = 64
test_loader = DataLoader(dataset, batch_size=64, shuffle=True)
onion = torch.load('cnn_model/output/train/model4.pth', map_location=torch.device('cpu')).to('cuda')
onion.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
losses = []
preds = []
labels = []
out_dir = 'cnn_model/output/test'
os.makedirs(out_dir, exist_ok=True)
for (input, regi, posi, info), label in tqdm(test_loader, desc="Testing"):
    input, regi, posi, label = input.to(device), regi.to(device), posi.to(device), label.to(device)
    pred = onion(input, regi, posi)
    loss = weighted_mse_loss(pred, label, 10)
    loss.backward()
    preds.append(pred.reshape(-1, r, z))
    labels.append(label.reshape(-1, r, z))
    losses.append(loss.item())

print(sum(losses) / len(losses))
json.dump(losses, open(f"{out_dir}/testing_loss.json", 'w'), indent=2)
preds = torch.concat(preds, dim=0)
labels = torch.concat(labels, dim=0)
json.dump(preds.tolist(), open(f"{out_dir}/preds.json", 'w'), indent=2)
json.dump(labels.tolist(), open(f"{out_dir}/labels.json", 'w'), indent=2)



