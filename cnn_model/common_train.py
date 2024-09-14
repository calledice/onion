import torch
from dataset import OnionDataset
from torch.utils.data import Dataset, DataLoader
from post_process import visualize
from onion_model import *
from common_predict import predict
import torch.nn as nn
from tqdm import tqdm
import os
import json
import numpy as np
import time
import random
import argparse

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
    os.makedirs(f'{out_dir}/train', exist_ok=True)
    min_val_loss = float('inf')
    loss_fn = nn.MSELoss()
    optim = torch.optim.Adam(params=model.parameters(), lr=config.lr)
    train_losses = []
    val_losses = []
    lambda_l1 = config.lambda_l1
    p = config.p
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        print(f"epoch: {epoch}")
        losses = []
        for (input, regi, posi, info), label in tqdm(train_loader, desc="Training"):
            input, regi, posi, label = input.to(device), regi.to(device), posi.to(device), label.to(device)
            if config.with_PI:
                pred = model(input, regi, posi)
                with open(f'{out_dir}/model_structure.txt', 'w') as f:
                    with redirect_stdout(f):
                        summary(model, input_data=[input, regi, posi])

            else:
                pred = model(input)
                with open(f'{out_dir}/model_structure.txt', 'w') as f:
                    with redirect_stdout(f):
                        summary(model, input_data=input)
            pred_temp = pred.unsqueeze(-1)
            result = torch.bmm(posi.view(len(posi), len(posi[0]), -1), pred_temp).squeeze(-1)

            optim.zero_grad()
            if config.addloss:
                loss_1 = loss_fn(pred, label)
                loss_2 = loss_fn(input, result)
                alpha = loss_1.item() / loss_2.item() if loss_2 > 0 else 10.0
                l1_reg = torch.tensor(0., device=device)
                for param in model.parameters():
                    l1_reg += torch.norm(param, p)
                loss = loss_1 + alpha * loss_2 + lambda_l1 * l1_reg
                # loss = weighted_mse_loss(pred, label, 10)
            else:
                loss = loss_fn(pred, label)
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
                alpha = loss_1.item() / loss_2.item() if loss_2 > 0 else 10.0
                l1_reg = torch.tensor(0., device=device)
                for param in model.parameters():
                    l1_reg += torch.norm(param, p)
                loss = loss_1 + alpha * loss_2 + lambda_l1 * l1_reg
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


def run(Module, config: Config):
    train_path = config.train_path
    val_path = config.val_path
    test_path = config.test_path
    out_dir = config.out_dir
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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
    print(f"max_n: {n}, max_r: {r}, max_z: {z}")
    onion = Module(n, r, z)
    onion.to(device)

    train_losses, val_losses = train(onion, train_loader, val_loader, config)
    plot_loss(train_losses, val_losses, out_dir)


def tmp_runner(dataset,Module, addloss = True ,predict_visualize=False, randomnumseed=None):
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
        add = "addloss"
    else:
        add = ""

    if Module == CNN_Base:
        out_dir = "../../onion_data/cnn_model/output/CNN_Base_input"
        with_PI = False
    elif Module == Onion_gavin:
        out_dir = "../../onion_data/cnn_model/output/Onion_gavin"
        with_PI = False
    elif Module == Onion_input:
        out_dir = f"../../onion_data/cnn_model/output/{name_dataset}_{randomnumseed}_Onion_input_{add}"
        with_PI = False
    elif Module == Onion_input_softplus:
        out_dir = f"../../onion_data/cnn_model/output/{name_dataset}_{randomnumseed}_Onion_input_{add}_softplus"
        with_PI = False
    elif Module == Onion_PI:
        out_dir = f"../../onion_data/cnn_model/output/{name_dataset}_{randomnumseed}_Onion_PI_{add}"
        with_PI = True
    elif Module == Onion_PI_softplus:
        out_dir = f"../../onion_data/cnn_model/output/{name_dataset}_{randomnumseed}_Onion_PI_{add}_softplus"
        with_PI = True
    # elif Module == Onion_PI_posiplus:
    #     out_dir = f"../../onion_data/cnn_model/output/{name_dataset}_Onion_PI_{add}_posiplus"
    #     with_PI = True
    # elif Module == Onion_PI_softplus_posiplus:
    #     out_dir = f"../../onion_data/cnn_model/output/{name_dataset}_Onion_PI_{add}_softplus_posiplus"
    #     with_PI = True
    elif Module == ResOnion_input:
        out_dir = f"../../onion_data/res_model/output/{name_dataset}_{randomnumseed}_ResOnion_input_{add}"
        with_PI = False
    elif Module == ResOnion_input_softplus:
        out_dir =  f"../../onion_data/res_model/output/{name_dataset}_{randomnumseed}_ResOnion_input_{add}_softplus"
        with_PI = False
    elif Module == ResOnion_PI:
        out_dir = f"../../onion_data/res_model/output/{name_dataset}_{randomnumseed}_ResOnion_PI_{add}"
        with_PI = True
    elif Module == ResOnion_PI_softplus:
        out_dir = f"../../onion_data/res_model/output/{name_dataset}_{randomnumseed}_ResOnion_PI_{add}_softplus"
        with_PI = True
    else:
        print("模型不在列表中")
        exit(1)
    print(f"with_PI: {with_PI} /n")
    print(f"addloss: {addloss} /n")
    config = Config(train_path, val_path, test_path, out_dir, with_PI, addloss, randomnumseed, early_stop=5, epochs=50,
                    batch_size=256, lambda_l1=0.0001, p=2)

    seed_everything(randomnumseed)

    print(out_dir)

    if predict_visualize:
        print("start predict")
        predict(config)
        print("start visualize")
        visualize(out_dir)
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
        visualize(out_dir)

if __name__ == '__main__':
    '''
    数据集路径和超参数设置均在tmp_runner函数中的config中设置
    '''
    parser = argparse.ArgumentParser(description='Train or predict with specified parameters.')
    parser.add_argument('--dataset',type = str, help='dataset name', default="phantom2A")
    parser.add_argument('--model', help='model name',default=Onion_input)
    parser.add_argument('--addloss', action='store_true', help='Add loss to training',default=False)
    parser.add_argument('--predict_visualize', action='store_true', help='Visualize predictions',default=False)
    parser.add_argument('--randomnumseed',type = int, help='Use random seed for reproducibility',default=42)
    args = parser.parse_args()

    # 调用 tmp_runner 函数并传入参数
    tmp_runner(dataset = args.dataset,
        Module=globals()[args.model],
               addloss=args.addloss,
               predict_visualize=args.predict_visualize,
               randomnumseed=args.randomnumseed)
    # tmp_runner(Onion_input, addloss=True, predict_visualize=False, randomnumseed=False)
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
    python common_train.py --dataset phantom2A --model Onion_input --randomnumseed 42
    
    ./phantom2A_tain.sh > ../../onion_data/phantom2A_training.log 2>&1 &
    ./phantomEAST_tain.sh > ../../onion_data/phantomEAST_training.log 2>&1 &
    ./EXP2A_tain.sh > ../../onion_data/EXP2A_training.log 2>&1 &
    ./EXPEAST_tain.sh > ../../onion_data/EXPEAST_training.log 2>&1 &
'''
