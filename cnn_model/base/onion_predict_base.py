import torch 
print(torch.cuda.is_available())
from dataset import OnionDataset
from onion_model import CNN_Base, weighted_mse_loss
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json
import os
print(torch.cuda.is_available())

dataset = OnionDataset("../data_Phantom/phantomdata/mini_1_test_database_1_100_100.h5")

# 临时加的，为了不做padding
n = int(dataset.input_len_org[0])
r = int(dataset.info_list[0][1])
z = int(dataset.info_list[0][2])

print(len(dataset))
batch_size = 64
out_dir = 'output/Phantom_base'
test_loader = DataLoader(dataset, batch_size=64, shuffle=True)
onion = torch.load(f'{out_dir}/train/model_best.pth', map_location=torch.device('cpu')).to('cuda')
onion.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
losses = []
preds = []
labels = []
os.makedirs(out_dir+'/test', exist_ok=True)
for (input, regi, posi, info), label in tqdm(test_loader, desc="Testing"):
    input, regi, posi, label = input.to(device), regi.to(device), posi.to(device), label.to(device)
    pred = onion(input)

    # 加权MSELoss
    loss = weighted_mse_loss(pred, label, 10)

    # 设置阈值为0.001，小于0.001的置为0
    pred = torch.where(pred < 0.001, torch.tensor(0.0), pred)
    print(loss)

    # 还原output形状
    preds.append(pred.reshape(-1, 32, 36))
    labels.append(label.reshape(-1, 32, 36))
    losses.append(loss.item())
    
    # break # 只测试了一个batch，如果要预测所有测试集则删除这个break

print(sum(losses) / len(losses))
json.dump(losses, open(f"{out_dir}/test/testing_loss.json", 'w'), indent=2)
preds = torch.concat(preds, dim=0)
labels = torch.concat(labels, dim=0)
json.dump(preds.tolist(), open(f"{out_dir}/test/preds.json", 'w'), indent=2)
json.dump(labels.tolist(), open(f"{out_dir}/test/labels.json", 'w'), indent=2)



