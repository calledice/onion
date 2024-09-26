import torch 
from dataset import OnionDataset
from onion_model import *
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json
import os
import torch.nn as nn
import numpy as np

# 获取当前源程序所在的目录
script_dir = os.path.dirname(os.path.abspath(__file__))

# 切换工作目录到源程序所在的目录
os.chdir(script_dir)

def predict(config: Config):
    dataset = OnionDataset(config.test_path)

    # 临时加的，为了不做padding
    n = int(dataset.input_len_org[0])
    r = int(dataset.info_list[0][1])
    z = int(dataset.info_list[0][2])
    
    print(len(dataset))
    batch_size = 64
    test_loader = DataLoader(dataset, batch_size=64, shuffle=False)
    out_dir = config.out_dir
    lambda_l2 = config.lambda_l2
    p = config.p
    device = config.device
    os.environ['CUDA_VISIBLE_DEVICES'] = config.device_num
    model = torch.load(f'{out_dir}/train/model_best.pth', map_location=torch.device('cuda'))
    model.eval()
    losses = []
    preds = []
    labels = []
    results = []
    label2results = []
    inputs = []
    os.makedirs(out_dir+'/test', exist_ok=True)
    loss_fn = nn.MSELoss()
    for (input, regi, posi, info), label in tqdm(test_loader, desc="Testing"):
        input, regi, posi, label = input.to(device), regi.to(device), posi.to(device), label.to(device)
        if config.with_PI:
            pred = model(input, regi, posi)
        else:
            pred = model(input)
        pred_temp = pred.unsqueeze(-1)
        result = torch.bmm(posi.view(len(posi), len(posi[0]), -1), pred_temp).squeeze(-1) #pre2results
        label2result = torch.bmm(posi.view(len(posi), len(posi[0]), -1), label.unsqueeze(-1)).squeeze(-1)
        if config.addloss:
            loss_1 = loss_fn(pred, label)
            loss_2 = loss_fn(input, result)
            alpha = loss_1.item() / loss_2.item() if loss_2 > 0 else 10.0
            l2_reg = torch.tensor(0., device=device)
            for param in model.parameters():
                l2_reg += torch.norm(param, p)
            loss = loss_1 + alpha * loss_2 + lambda_l2 * l2_reg
        else:
            loss = loss_fn(pred, label)

        # 设置阈值为0.001，小于0.001的置为0
        # pred = torch.where(pred < 0.001, torch.tensor(0.0), pred)
        # print(pred[0])

        # 还原output形状
        preds.append(pred.detach().reshape(-1, z, r))
        labels.append(label.detach().reshape(-1, z, r))
        losses.append(loss.item())
        results.append(result.detach())
        label2results.append(label2result.detach())
        inputs.append(input.detach())
        
        # break # 只测试了一个batch，如果要预测所有测试集则删除这个break

    print(sum(losses) / len(losses))
    json.dump(losses, open(f"{out_dir}/test/testing_loss.json", 'w'), indent=2)
    preds = torch.concat(preds, dim=0)
    labels = torch.concat(labels, dim=0)
    results = torch.concat(results, dim=0)
    label2results = torch.concat(label2results, dim=0)
    inputs = torch.concat(inputs, dim=0)
    json.dump(preds.tolist(), open(f"{out_dir}/test/preds.json", 'w'), indent=2)
    json.dump(labels.tolist(), open(f"{out_dir}/test/labels.json", 'w'), indent=2)
    json.dump(results.tolist(), open(f"{out_dir}/test/results.json", 'w'), indent=2)
    json.dump(label2results.tolist(), open(f"{out_dir}/test/label2results.json", 'w'), indent=2)
    json.dump(inputs.tolist(), open(f"{out_dir}/test/inputs.json", 'w'), indent=2)
    print("finish")