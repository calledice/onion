import os
from tqdm import tqdm, trange
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import OnionDataset
from onion_model_crossatten import Onion, Config, ConvEmbModel
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from torch.utils.tensorboard import SummaryWriter
import argparse
import json
import d2l

outpath = './model_data/'
data_root = './data_HL_2A'
model_ckpt = None
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = Onion
os.makedirs(outpath, exist_ok=True)

def draw_loss(train_loss_list, val_loss_list, out_path):
    train_epoch_list = [el[0] for el in train_loss_list]
    train_loss_list = [el[1] for el in train_loss_list]
    val_epoch_list = [el[0] for el in val_loss_list]
    val_loss_list = [el[1] for el in val_loss_list]
    plt.figure(figsize=(5, 4))
    plt.title("loss during training")  # 标题
    plt.xlabel("iter")  # Changed "iter" to "Epoch" assuming epochs are being plotted
    plt.ylabel("Loss")  # Adding a ylabel for clarity
    plt.plot(train_epoch_list, train_loss_list, label="train_loss")
    plt.plot(val_epoch_list, val_loss_list, label="valid_loss", marker="+")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid()
    plt.savefig(out_path)
    plt.close()


def training(model_name, config, train_input_path,val_input_path, num_train_epochs,
             weight_decay, learning_rate, scheduler_step, check_every, out_path,
             tb_save_path):
    print("start training ")
    max_input_len, max_rz_len = config.max_input_len, config.max_rz_len
    train_set = OnionDataset(train_input_path, max_input_len=max_input_len, max_rz_len=max_rz_len)
    train_iter = DataLoader(train_set, batch_size=config.batch_size, drop_last=True, shuffle=True)
    valid_set= OnionDataset(val_input_path, max_input_len=max_input_len, max_rz_len=max_rz_len)
    val_iter = DataLoader(valid_set, batch_size=config.batch_size, drop_last=True, shuffle=True)
    num_gpus = 2
    devices = [d2l.try_gpu(i) for i in range(num_gpus)]
    model = Onion(config)
    model = nn.DataParallel(model,device_ids=devices)
    out_path = os.path.join(out_path, model_name + time.strftime("%Y-%m-%d-%X", time.localtime()))
    log_path = os.path.join(out_path, "result.txt")
    tb_save_path = os.path.join(out_path, tb_save_path)

    os.makedirs(out_path, exist_ok=True)
    os.makedirs(tb_save_path, exist_ok=True)

    train_model(model, model_name, num_train_epochs, weight_decay, learning_rate, scheduler_step, train_iter, val_iter,
                check_every,
                out_path, tb_save_path, log_path, config,devices)


def train_model(model, model_name, num_train_epochs, weight_decay, learning_rate, scheduler_step, train_iter, val_iter,
                check_every,
                out_path, tb_save_path, log_path, config,devices):
    max_input_len, max_rz_len = config.max_input_len, config.max_rz_len
    loss_mse = nn.MSELoss()
    conv_emb = ConvEmbModel(out_channels=max_rz_len)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_train_epochs,
                                                           eta_min=0, last_epoch=-1, verbose=False)
    writer = SummaryWriter(tb_save_path)

    global_step = 0
    train_loss = 0.0
    train_loss_list = []
    val_loss_list = []
    step = 0
    model.zero_grad()
    for epoch in range(num_train_epochs):
        train_loss_epoch = []
        model.train()
        epoch_iterator = tqdm(train_iter, desc=f"Epoch {epoch}", mininterval=2)

        for batch_id, ((input, regi, posi, info), label) in enumerate(epoch_iterator):
            input, regi, posi, label= input.to(devices[0]), regi.to(devices[0]), posi.to(devices[0]), label.to(devices[0])
            output = model(input, regi, posi).squeeze(1)
            output_b = output.unsqueeze(-1)#
            result = torch.bmm(posi, output_b).squeeze(-1)#
            sigmoid_n = torch.sigmoid(model.n)
            loss = loss_mse(output, label)+sigmoid_n *loss_mse(input,result)# 怎么增加可学习的权重？
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_value = loss.item()
            train_loss += loss_value
            train_loss_list.append((step, loss_value))
            train_loss_epoch.append(loss_value)
            writer.add_scalar("train_loss", loss_value, step)
            step += 1

            if step % check_every == 0:
                val_loss = evaluate(model, model_name, val_iter, loss_mse)
                val_loss_list.append((step - 1, val_loss))
                writer.add_scalar("val_loss", val_loss, step, double_precision=True)
                with open(log_path, "a") as f:
                    f.write("\n epoch:{}, val_loss:{:.10f}".format(step, val_loss))

        scheduler.step()
        print(f'sigmoid_n = {sigmoid_n}')
        epoch_loss = sum(train_loss_epoch) / len(train_loss_epoch)
        print("\ntrain_loss------>:", epoch_loss)

        if (epoch + 1) % 5 == 0:
            with open(log_path, "a") as f:
                f.write("\n epoch:{}, train_loss:{:.10f}".format(epoch, epoch_loss))
                # model_params_path = os.path.join(out_path, model_name + "_params" + str(epoch) + ".pth")
                model_path = os.path.join(out_path, model_name + "_" + str(epoch) + ".pth")
                # torch.save(model.state_dict(), model_params_path)
                torch.save(model, model_path)

    loss_str = ["step {}: {}".format(*el) for el in train_loss_list]
    with open(os.path.join(out_path, "train_loss.txt"), "w") as fw:
        fw.write("\n".join(loss_str))

    # 定义输出配置文件路径
    output_json = f"config_and_args_{config.n_layer}L_norm_2loss_{config.max_rz_len}.json"
    output_file = os.path.join(out_path,  output_json)
    # 使用 json 保存
    data_to_dump = {
    'args': args.__dict__,
    'config': config_dict
}
    with open(output_file, 'w') as f:
        json.dump(data_to_dump, f, indent=4)
    with open(out_path+'/'+'model.txt','a') as file0:
        print(model,file=file0)
    output_fig = f"{config.n_layer}L_norm_2loss_{config.max_rz_len}.png"
    output_path = os.path.join(out_path, output_fig)
    draw_loss(train_loss_list, val_loss_list, out_path)


def evaluate(model, model_name, val_iter, loss_mse):
    print('\nevaluating')
    ### 交叉验证
    val_loss = 0.0
    val_loss_list = []
    conv_emb = ConvEmbModel(out_channels=config.max_rz_len)
    model.eval()
    with torch.no_grad():
        for batch_id, ((input, regi, posi, info), label) in enumerate(val_iter):
            input, regi, posi, label = input.to(device), regi.to(device), posi.to(device), label.to(device)
            # output = model(input, regi, posi).squeeze(1).to(torch.float32)
            output = model(input, regi, posi).squeeze(1)
            output_b = output.unsqueeze(-1)  #
            result = torch.bmm(posi, output_b).squeeze(-1)  #
            sigmoid_n = torch.sigmoid(model.n)
            loss = loss_mse(output, label) + sigmoid_n * loss_mse(input, result)
            val_loss += loss.to(device).item()
            val_loss_list.append(val_loss)
        val_loss = sum(val_loss_list) / len(val_loss_list)
        print("\nval_loss----->:", val_loss)
    model.train()

    return val_loss


if __name__ == "__main__":
    paser = argparse.ArgumentParser()
    # paser.add_argument("--model_name", help="选择模型", default="expert_mmoe")
    paser.add_argument("--model_name", help="选择模型", default="Onion")
    paser.add_argument("--train_input_path", help="训练集输入数据路径", default="./data_Phantom/phantomdata/mini_1_train_database.h5")
    paser.add_argument("--val_input_path", help="验证集输入数据路径", default="./data_Phantom/phantomdata/mini_1_valid_database.h5")
    paser.add_argument("--num_train_epochs", help="num_train_epochs", type=int, default=10)
    paser.add_argument("--weight_decay", help="weight_decay", type=float, default=0.005)
    paser.add_argument("--learning_rate", help="learning_rate", type=float, default=5e-4)
    paser.add_argument("--scheduler_step", help="lr更新步长", type=int, default=500)
    paser.add_argument("--check_every", help="每多少步validate一次", type=int, default=2000)
    # paser.add_argument("--out_path", help="输出路径", default="./model_data")
    paser.add_argument("--out_path", help="输出路径", default="./model_attn_data")
    paser.add_argument("--tb_save_path", help="TensorBoard 保存路径", default="TensorBoard_logs")
    args = paser.parse_args()

    config = Config(4, 8, 0.0, True, torch.float32, 4,100,2048)
    config_dict = {
        "n_layer": config.n_layer,
        "n_head": config.n_head,
        "dropout": config.dropout,
        "bias": config.bias,
        "dtype": str(config.dtype),
        "batch_size": config.batch_size,
        "max_input_len": config.max_input_len,
        "max_rz_len": config.max_rz_len
    }


    st = time.time()
    training(model_name=args.model_name,
             config=config,
             train_input_path=args.train_input_path,
             val_input_path=args.val_input_path,
             num_train_epochs=args.num_train_epochs,
             weight_decay=args.weight_decay,
             learning_rate=args.learning_rate,
             scheduler_step=args.scheduler_step,
             check_every=args.check_every,
             out_path=args.out_path,
             tb_save_path=args.tb_save_path)
    te = time.time()
    print("time: ", te - st)
