import torch 
from dataset import OnionDataset
from onion_model import *
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json
import os
import torch.nn as nn
import numpy as np
from post_process import visualize_up
from pathlib import Path

# 获取当前源程序所在的目录
script_dir = os.path.dirname(os.path.abspath(__file__))

# 切换工作目录到源程序所在的目录
os.chdir(script_dir)

def predict(config: Config):
    dataset = OnionDataset(config.test_path)

    # 临时加的，为了不做padding
    # n = int(dataset.input_len_org)
    r = int(dataset.info_list[0][1])
    z = int(dataset.info_list[0][2])
    
    print(len(dataset))
    batch_size = 64
    test_loader = DataLoader(dataset, batch_size=64, shuffle=False)
    out_dir = config.out_dir
    lambda_l2 = config.lambda_l2
    p = config.p
    device = config.device
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    model = globals()[config.Module]
    config.device_num = "0"
    # 加载模型
    # 加载状态字典
    state_dict = torch.load(f'{out_dir}/train/model_best.pth', map_location=torch.device('cuda'))

    # 加载状态字典到模型
    model.load_state_dict(state_dict)
    # model = torch.load(f'{out_dir}/train/model_best.pth', map_location=torch.device('cuda'))

    # # 如果模型是 DDP 包装的，去除 DDP 包装
    # if isinstance(model, torch.nn.parallel.DistributedDataParallel):
    #     model = model.module


    model.eval()
    losses = []
    preds = []
    labels = []
    results = []
    label2results = []
    inputs = []
    out_dir_file = out_dir+'/test_new'
    os.makedirs(out_dir_file, exist_ok=True)
    loss_fn = nn.MSELoss()
    for input, posi,posi_origin, label in tqdm(test_loader, desc="Testing"):
        input,  posi, posi_origin, label = input.to(device), posi.to(device),posi_origin.to(device), label.to(device)
        if config.with_PI:
            pred = model(input, posi)
        else:
            pred = model(input)
        pred_temp = pred.unsqueeze(-1)
        result = torch.bmm(posi_origin.view(len(posi), len(posi[0]), -1), pred_temp).squeeze(-1) #pre2results
        label2result = torch.bmm(posi_origin.view(len(posi), len(posi[0]), -1), label.unsqueeze(-1)).squeeze(-1)
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
    json.dump(losses, open(f"{out_dir_file}/testing_loss.json", 'w'), indent=2)
    preds = torch.concat(preds, dim=0).tolist()
    labels = torch.concat(labels, dim=0).tolist()
    results = torch.concat(results, dim=0).tolist()
    label2results = torch.concat(label2results, dim=0).tolist()
    inputs = torch.concat(inputs, dim=0).tolist()

    visualize_up(preds, labels, inputs, results, label2results, config.out_dir)

    json.dump(preds[:1000], open(f"{out_dir_file}/preds_1000.json", 'w'), indent=2)
    json.dump(labels[:1000], open(f"{out_dir_file}/labels_1000.json", 'w'), indent=2)
    json.dump(results, open(f"{out_dir_file}/results.json", 'w'), indent=2)
    json.dump(label2results, open(f"{out_dir_file}/label2results.json", 'w'), indent=2)
    json.dump(inputs, open(f"{out_dir_file}/inputs.json", 'w'), indent=2)
    json.dump(config.as_dict(), open(f"{out_dir_file}/config.json", 'w'), indent=4)
    print("finish")


if __name__ == "__main__":
    print("start")
    case_path = "../../onion_train_data/train_results_EAST/train_results_EAST-0.2/"
    # case_path = "../../onion_train_data/train_results_2A/"
    ###########################  遍历文件夹时用
    # # 创建Path对象
    # path = Path(case_path)
    # # 获取该层级的所有文件夹，不会递归到子文件夹中
    # folders = [f.name for f in path.iterdir() if f.is_dir()]
    # # folders = folders[17:]
    # for file_name in folders:
    #     config_path = case_path+file_name+"/config.json"
    #     # 使用 json 模块加载 JSON 文件
    #     with open(config_path, 'r') as file:
    #         info = json.load(file)
    #     # info_list = list(info.values())[0]
    #     config = Config(**info,randomnumseed=42)
    #     config.out_dir = case_path + file_name
    #     predict(config)
        ############################# 单个时用
    file_name = "phantomEAST-0.2_42_Onion_PI_uptime_adam_scheduler"
    config_path = case_path+file_name+"/config.json"
    # 使用 json 模块加载 JSON 文件
    with open(config_path, 'r') as file:
        info = json.load(file)
    # info_list = list(info.values())[0]
    config = Config(**info,randomnumseed=42)
    config.out_dir = case_path + file_name
    predict(config)