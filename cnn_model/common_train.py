import torch
from dataset import OnionDataset
from torch.utils.data import Dataset, DataLoader
from post_process import visualize
from onion_model import Onion_input,Onion_PI,Onion_PI_up,ResOnion_input,ResOnion_PI,ResOnion_PI_up,Onion_input_softplus,Onion_PI_softplus,ResOnion_input_softplus,ResOnion_PI_softplus,Config
from common_predict import predict
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn as nn
from tqdm import tqdm
import math
import os
import json
import numpy as np
import time
import random
import argparse
from contextlib import redirect_stdout
from torchinfo import summary

# 获取当前源程序所在的目录
script_dir = os.path.dirname(os.path.abspath(__file__))

# 切换工作目录到源程序所在的目录
os.chdir(script_dir)

# 固定随机种子
def seed_everything(seed=42):
    random.seed(seed)  # 为Python内置的random模块设置随机种子
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为Python的哈希行为设置随机种子
    np.random.seed(seed)  # 为numpy设置随机种子
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
    torch.backends.cudnn.deterministic = True  # 设置cuDNN为确定性模式
    torch.backends.cudnn.benchmark = False  # 禁止使用cuDNN的benchmark功能


def train(model, train_loader, val_loader, config: Config):
    '''
    with_PI: 判断是不是没有regi和posi的模型
    '''
    out_dir = config.out_dir
    epochs = config.epochs
    early_stop = config.early_stop
    device = config.device
    alfa = config.alfa
    os.makedirs(f'{out_dir}/train', exist_ok=True)
    min_val_loss = float('inf')
    loss_fn = nn.MSELoss()
    optim = torch.optim.Adam(params=model.parameters(), lr=config.lr)
    # 创建余弦退火学习率调度器
    T_max = 50  # 一个周期的长度
    scheduler = CosineAnnealingLR(optim, T_max=T_max, eta_min=0.00001)
    train_losses = []
    val_losses = []
    lambda_l2 = config.lambda_l2
    p = config.p
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        print(f"epoch: {epoch}")
        losses = []
        for (input, regi, posi, info), label in tqdm(train_loader, desc="Training"):
            # plt.imshow(np.array(label[0].reshape(36, 32)), cmap='gray', interpolation='nearest')
            # # 显示颜色条
            # plt.colorbar()
            # plt.show()
            input, regi, posi, label = input.to(device), regi.to(device), posi.to(device), label.to(device)
            if config.with_PI:
                pred = model(input, regi, posi)
                if epoch == 0:
                    with open(f'{out_dir}/model_structure.txt', 'w') as f:
                        with redirect_stdout(f):
                            summary(model, input_data=[input, regi, posi])

            else:
                pred = model(input)
                if epoch == 0:
                    with open(f'{out_dir}/model_structure.txt', 'w') as f:
                        with redirect_stdout(f):
                            summary(model, input_data=input)
            pred_temp = pred.unsqueeze(-1)
            result = torch.bmm(posi.view(len(posi), len(posi[0]), -1), pred_temp).squeeze(-1)

            optim.zero_grad()
            if config.addloss:
                loss_1 = loss_fn(pred, label)
                loss_2 = loss_fn(input, result)
                alpha = alfa * loss_1.item() / loss_2.item() if loss_2 > 0 else 10.0 #0.618
                beta = 1.0
                # alpha = loss_1.item() / (loss_1.item() + loss_2.item())
                # beta = loss_2.item() / (loss_1.item() + loss_2.item())
                l2_reg = torch.tensor(0., device=device)
                for param in model.parameters():
                    l2_reg += torch.norm(param, p)
                loss = beta * loss_1 + alpha * loss_2 + lambda_l2 * l2_reg
            else:
                loss = loss_fn(pred, label)
            loss.backward()
            optim.step()
            losses.append(loss.item())
        scheduler.step()
        # 打印当前的学习率
        current_lr = optim.param_groups[0]['lr']
        print(f'Epoch {epoch + 1}: Learning Rate = {current_lr}')
        train_loss = sum(losses) / len(losses)
        print(f"epoch{epoch} training loss: {train_loss}\n")
        current_lr = scheduler.get_last_lr()[0]
        print(f"epoch{epoch} Learning Rate: {current_lr}\n")
        json.dump(losses, open(f"{out_dir}/train/training_loss{epoch}.json", 'w'))
        train_losses.append(train_loss)

        # 验证阶段
        model.eval()
        best_epoch = -1
        losses = []
        preds = []
        labels = []
        loss_fn = nn.MSELoss()
        for (input, regi, posi, info), label in tqdm(val_loader, desc="Validating"):
            input, regi, posi, label = input.to(device), regi.to(device), posi.to(device), label.to(device)
            if config.with_PI:
                pred = model(input, regi, posi)
            else:
                pred = model(input)
            pred_temp = pred.unsqueeze(-1)
            result = torch.bmm(posi.view(len(posi), len(posi[0]), -1), pred_temp).squeeze(-1)
            if config.addloss:
                loss_1 = loss_fn(pred, label)
                loss_2 = loss_fn(input, result)
                alpha = alfa * loss_1.item() / loss_2.item() if loss_2 > 0 else 10.0 #0.618
                beta = 1.0
                # beta = loss_1.item() / (loss_1.item() + loss_2.item())
                # alpha = loss_2.item() / (loss_1.item() + loss_2.item())
                l2_reg = torch.tensor(0., device=device)
                for param in model.parameters():
                    l2_reg += torch.norm(param, p)
                loss = beta * loss_1 + alpha * loss_2 + lambda_l2 * l2_reg
            else:
                loss = loss_fn(pred, label)
            preds.append(pred.detach().reshape(-1, config.max_r, config.max_z))
            labels.append(label.detach().reshape(-1, config.max_r, config.max_z))
            losses.append(loss.detach().item())
        val_loss = sum(losses) / len(losses)
        
        print(f"epoch{epoch} min loss: {min_val_loss}")
        print(f"epoch{epoch} validation loss: {val_loss}")

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save(model, f"{out_dir}/train/model_best.pth")
            with open(f"{out_dir}/train/best_epoch.txt", 'w') as f:
                f.write(str(epoch))
            if early_stop > 0:
                early_stop = config.early_stop
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
    plt.xlabel('epochs')
    plt.ylabel('loss')
    # 显示图例
    plt.legend()
    plt.savefig(f'{out_dir}/train/loss_curve.png')
    # 显示图形
    plt.show()


def run(Module, config: Config):
    train_path = config.train_path
    val_path = config.val_path
    test_path = config.test_path
    out_dir = config.out_dir
    device = config.device
    os.environ['CUDA_VISIBLE_DEVICES'] = config.device_num

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
    print(f"max_n: {n}, max_r: {r}, max_z: {z}")
    onion = Module(n, r, z)
    onion.to(device)

    train_losses, val_losses = train(onion, train_loader, val_loader, config)
    plot_loss(train_losses, val_losses, out_dir)


def tmp_runner(dataset,Module, addloss = True ,predict_visualize=False, randomnumseed=None,lr = 0.0001,device_num = "0",alfa=1.0):
    name_dataset = dataset
    if name_dataset == "phantom2A":
        train_path = "../data_Phantom/phantomdata/HL-2A_train_database_1_100_1000.h5"
        val_path = "../data_Phantom/phantomdata/HL-2A_valid_database_1_100_1000.h5"
        test_path = "../data_Phantom/phantomdata/HL-2A_test_database_1_100_1000.h5"
    elif name_dataset == "phantomEAST":
        train_path = "../data_Phantom/phantomdata/EAST_train_database_1_100_1000.h5"
        val_path = "../data_Phantom/phantomdata/EAST_valid_database_1_100_1000.h5"
        test_path = "../data_Phantom/phantomdata/EAST_test_database_1_100_1000.h5"
    elif name_dataset == "EXP2A":
        train_path = "../data_HL_2A/data/HL_2A_train_database.h5"
        val_path = "../data_HL_2A/data/HL_2A_val_database.h5"
        test_path = "../data_HL_2A/data/HL_2A_test_database.h5"
    elif name_dataset == "EXPEAST":
        train_path = "../data_East/data/EAST_train_database.h5"
        val_path = "../data_East/data/EAST_valid_database.h5"
        test_path = "../data_East/data/EAST_test_database.h5"
    else:
        print("dataset is not included")

    if addloss:
        add = "addloss"+str(alfa)
    else:
        add = ""
    out_root_path = "../../onion_data/model_train_0.0001/output-50/"

    if Module == Onion_input:
        out_dir = out_root_path+f"{name_dataset}_{randomnumseed}_Onion_input_{add}"
        with_PI = False
    elif Module == Onion_input_softplus:
        out_dir = out_root_path+f"{name_dataset}_{randomnumseed}_Onion_input_{add}_softplus"
        with_PI = False
    elif Module == Onion_PI:
        out_dir = out_root_path+f"{name_dataset}_{randomnumseed}_Onion_PI_{add}"
        with_PI = True
    elif Module == Onion_PI_up:
        out_dir = out_root_path+f"{name_dataset}_{randomnumseed}_Onion_PI_up_{add}"
        with_PI = True
    elif Module == Onion_PI_softplus:
        out_dir = out_root_path+f"{name_dataset}_{randomnumseed}_Onion_PI_{add}_softplus"
        with_PI = True
    elif Module == ResOnion_PI_up:
        out_dir = out_root_path + f"{name_dataset}_{randomnumseed}_ResOnion_PI_up_{add}"
        with_PI = True
    elif Module == ResOnion_input:
        out_dir = out_root_path+f"{name_dataset}_{randomnumseed}_ResOnion_input_{add}"
        with_PI = False
    elif Module == ResOnion_input_softplus:
        out_dir =  out_root_path+f"{name_dataset}_{randomnumseed}_ResOnion_input_{add}_softplus"
        with_PI = False
    elif Module == ResOnion_PI:
        out_dir = out_root_path+f"{name_dataset}_{randomnumseed}_ResOnion_PI_{add}"
        with_PI = True
    elif Module == ResOnion_PI_softplus:
        out_dir = out_root_path+f"{name_dataset}_{randomnumseed}_ResOnion_PI_{add}_softplus"
        with_PI = True
    else:
        print("模型不在列表中")
        exit(1)
    print(f"with_PI: {with_PI} /n")
    print(f"addloss: {addloss} /n")
    config = Config(train_path, val_path, test_path, out_dir, with_PI, addloss, randomnumseed, early_stop=-1, epochs=50,
                    batch_size=256, lambda_l2=0.0001, p=2, lr=lr,device_num=device_num,alfa=alfa)

    seed_everything(randomnumseed)

    print(out_dir)

    if predict_visualize:
        print("start predict")
        predict(config)
        # print("start visualize")
        # visualize(out_dir)
    else:
        print("start train")
        # 记录训练开始时间
        start_time = time.time()
        run(Module, config)
        # 记录训练结束时间
        end_time = time.time()
        # 计算训练总耗时
        training_time = (end_time - start_time)/60
        with open(f"{out_dir}/train/best_epoch.txt", 'a') as f:
            f.write(f"training time:{training_time} min \n")
        print(f"Total training time: {training_time:.2f} mins")
        predict(config)
        # visualize(out_dir)

if __name__ == '__main__':
    '''
    数据集路径和超参数设置均在tmp_runner函数中的config中设置
    '''
    parser = argparse.ArgumentParser(description='Train or predict with specified parameters.')
    parser.add_argument('--dataset',type = str, help='dataset name', default="EXP2A")
    parser.add_argument('--model', help='model name',default=Onion_PI_up)
    parser.add_argument('--addloss', action='store_true', help='Add loss to training',default=False)
    parser.add_argument('--pv', action='store_true', help='Visualize predictions',default=False)
    parser.add_argument('--randomnumseed',type = int, help='Use random seed for reproducibility',default=42)
    parser.add_argument('--lr',type = float, help='learning rate',default=0.0001)
    parser.add_argument('--alfa',type = float, help='co_loss2',default=1.0)
    parser.add_argument('--device_num',type = str, help='device',default="0")
    args = parser.parse_args()
    args.model = globals()[args.model]

    # 调用 tmp_runner 函数并传入参数
    tmp_runner(dataset = args.dataset,
            Module=args.model,
            addloss=args.addloss,
            predict_visualize=args.pv,
            randomnumseed=args.randomnumseed,
            lr = args.lr,
            device_num = args.device_num,
            alfa=args.alfa)
    # tmp_runner(dataset="EXP2A" ,Module=ResOnion_PI_up, addloss=False, predict_visualize=False, randomnumseed=42)
'''
    dataset name: phantom2A  phantomEAST  EXP2A  EXPEAST
    model name:  
        Onion_input
        Onion_input_softplus
        Onion_PI
        Onion_PI_softplus
        ResOnion_input
        ResOnion_input_softplus
        ResOnion_PI
        ResOnion_PI_softplus
    python common_train.py --dataset EXP2A --model ResOnion_PI_up
    
      nohup ./phantom2A_train.sh > ../../onion_data/model_train_0.0001/phantom2A_training-50-all.log 2>&1 & 2878317
      nohup ./phantomEAST_train.sh > ../../onion_data/model_train_f/phantomEAST_training-50-1.log 2>&1 & 3921141
      nohup ./EXP2A_train.sh > ../../onion_data/model_train_0.0001/EXP2A_training-50-2.log 2>&1 &
      nohup ./EXPEAST_train.sh > ../../onion_data/model_train_f/EXPEAST_training-50.log 2>&1 &
'''