import torch
from dataset import OnionDataset
from torch.utils.data import Dataset, DataLoader
from onion_model import CNN_Base, weighted_mse_loss, Onion, OnionWithoutRegi, Config
from common_predict import predict
import torch.nn as nn
from tqdm import tqdm
import os
import json

# 获取当前源程序所在的目录
script_dir = os.path.dirname(os.path.abspath(__file__))

# 切换工作目录到源程序所在的目录
os.chdir(script_dir)


def train(model, train_loader, val_loader, config:Config):
    '''
    no_regi: 判断是不是没有regi和posi的模型
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
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        print(f"epoch: {epoch}")
        losses = []
        for (input, regi, posi, info), label in tqdm(train_loader, desc="Training"):
            input, regi, posi, label = input.to(device), regi.to(device), posi.to(device), label.to(device)
            if config.no_regi:
                pred = model(input)
            else:
                pred = model(input, regi, posi)
            optim.zero_grad()
            # loss = weighted_mse_loss(pred, label, 10)
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
        for (input, regi, posi, info), label in tqdm(val_loader, desc="Validating"):
            input, regi, posi, label = input.to(device), regi.to(device), posi.to(device), label.to(device)
            if config.no_regi:
                pred = model(input)
            else:
                pred = model(input, regi, posi)
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

def run(Module, config:Config):
    train_path = config.train_path
    val_path = config.val_path
    test_path = config.test_path
    out_dir = config.out_dir

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
    print(f"max_n: {n}, max_r: {r}, max_z: {z}")
    onion = Module(n, r, z)
    onion.to(device)

    train_losses, val_losses = train(onion, train_loader, val_loader, config)
    plot_loss(train_losses, val_losses, out_dir)

def tmp_runner(Module, predict_only=False):
    train_path = "../data_Phantom/phantomdata/mini_1_train_database_1_100_100.h5"
    val_path = "../data_Phantom/phantomdata/mini_1_valid_database_1_100_100.h5"
    test_path = "../data_Phantom/phantomdata/mini_1_test_database_1_100_100.h5"

    if Module == CNN_Base:
        out_dir = "output/Phantom_base"
        no_regi=True
    elif Module == Onion:
        out_dir = "output/Phantom_PI"
        no_regi=False
    elif Module == OnionWithoutRegi:
        out_dir = "output/Phantom"
        no_regi=True
    else:
        print("目前只支持CNN_Base, Onion, OnionWithoutRegi这三个模型")
        exit(1)

    config = Config(train_path, val_path, test_path, out_dir, no_regi, early_stop=-1, epochs=5, batch_size=64)
    
    if not predict_only:
        run(Module, config)
    predict(config)


if __name__ == '__main__':
    '''
    对于已经开发好的三个模型，直接通过这一个common_train文件就可以开启训练和预测，如果只需要预测，则开启predict_only=True.
    数据集路径和超参数设置均在tmp_runner函数中的config中设置
    '''
    tmp_runner(Onion, predict_only=True)