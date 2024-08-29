import os
import os.path as osp
from tqdm import tqdm, trange
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
from model import Onion, Config, region_embedding_func, position_embedding_func
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from torch.utils.tensorboard import SummaryWriter
import argparse
import h5py

outpath = 'model_data/'
data_root = './data_HL_2A'
model_ckpt = None
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = Onion
os.makedirs(outpath, exist_ok=True)


def get_dataset_old(pth_):
    # inp_pth = data_root + '/' + name + '_inputs_0604.csv'
    # out_pth = data_root + '/' + name + '_outputs_0604.csv'
    # inp_pth_ = data_root + '/' + name + '_inputs_0604.h5'
    # out_pth_ = data_root + '/' + name + '_outputs_0604.h5'

    # df_inp = pd.read_csv(inp_pth, header=None)
    # df_inp.to_hdf(inp_pth_,'table')
    # df_out = pd.read_csv(out_pth, header=None)PerformanceWarning:
    # your performance may suffer as PyTables will pickle object types that it cannot
    # map directly to c-types [inferred_type->mixed-integer,key->axis0] [items->None]
    # df_out.to_hdf(out_pth_, 'table')
    # database = pd.read_hdf(pth_, "table").drop(0)
    with h5py.File(pth_, 'r') as database:
        # input = np.array(pd.read_hdf(pth_, "table").drop(0))
        # input = input[:,:-1]
        # df_inp = df_inp.drop(df_inp.columns[0], axis=1)
        # output = np.array(pd.read_hdf(pth_, "table").drop(0))
        # input = np.array([])
        x = database["x"]
        y = database["y"]
        inputs_list = []  # 初始化一个列表来收集数据集
        outputs_list = []
        regi_list = []
        posi_list = []
        # 初始化一个列表来收集数据集
        for dkey in x.keys():
            inputs_list.append(x[dkey][:])  # 使用[:]确保数据被复制到列表中
            outputs_list.append(y[dkey][:])
        input = np.array([np.array(inputs_i) for inputs_i in inputs_list])
        output = np.array(outputs_list)

        regi = database['regi']
        posi = database['posi']
        for dkey in regi.keys():
            regi_list.append(regi[dkey][:])  # 使用[:]确保数据被复制到列表中
            posi_list.append(posi[dkey][:])
        regi = np.array(regi_list)
        posi = np.array(posi_list)

        input = torch.from_numpy(input).to(torch.float)
        output = torch.from_numpy(output).to(torch.float)
        dataset = TensorDataset(input, output)
    return dataset, regi, posi

def pad_arrays_to_length(arrays, target_length=101, label = 0):
    """
    将列表中的数组padding到指定长度，并记录原始长度。
    参数:
    - arrays: 不同长度数组的列表
    - target_length: 目标padding长度，默认为101
    返回:
    - padded_arrays: 所有数组padding到相同长度后的列表
    - original_lengths: 每个数组的原始长度列表
    """
    padded_arrays = []
    original_lengths = []
    for arr in arrays:
        # 计算需要padding的长度
        pad_length = target_length - arr.shape[1]
        # 确保padding长度非负，如果原始长度大于目标长度，不进行padding
        if pad_length > 0:
            # 使用0进行padding，默认在数组末尾添加
            padded_arr = np.pad(arr[0], (0, pad_length), mode='constant')
        else:
            # 如果原始长度超过目标长度，这里可以选择截断或者保持原样（取决于需求）
            padded_arr = arr[1][:target_length]  # 这里选择截断
        if label == 1 :
            padded_arr[-1] = arr.shape[1]
        padded_arrays.append(padded_arr.reshape((1,len(padded_arr))))
        original_lengths.append(arr.shape[1])  # 记录原始长度

    return padded_arrays

class CustomDataset(Dataset):
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
        # self.regi = regi
        # self.posi = posi
    def __getitem__(self, idx):
        input_seq = torch.tensor(self.inputs[idx], dtype=torch.float32)
        output = torch.tensor(self.outputs[idx].tolist(), dtype=torch.float32)
        # regi_info = torch.tensor(self.regi[idx], dtype=torch.float32)
        # posi_info = torch.tensor(self.posi[idx], dtype=torch.float32)
        # return input_seq, output, regi_info, posi_info  # 返回输入序列、输出及附加信息
        return input_seq, output  # 返回输入序列、输出及附加信息
    def __len__(self):
        return len(self.inputs)

def get_dataset(pth_):
    with h5py.File(pth_, 'r') as database:
        x = database["x"]
        y = database["y"]
        regi = database['regi']
        posi = database['posi']
        inputs_list = [x[dkey][:] for dkey in x.keys()]  # 收集输入数据
        outputs_list = [y[dkey][:].T for dkey in
                        y.keys()]  # 收集输出数据
        #todo 这里pad有点多余
        inputs_list_pad = pad_arrays_to_length(inputs_list,label=1)
        outputs_list_pad = pad_arrays_to_length(outputs_list,config.n_embd,label=0)
        regi_list = [regi[dkey][:] for dkey in regi.keys()]  # 收集regi信息
        posi_list = [posi[dkey][:] for dkey in posi.keys()]  # 收集posi信息
    # 直接使用列表，不再转为numpy数组然后转为torch.Tensor，因为我们要保留变长序列特性
    dataset = CustomDataset(inputs_list_pad, outputs_list_pad)
    return dataset,regi_list,posi_list


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
    plt.ylim(0, 0.1)
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(out_path, "loss-6L-norm-2loss.png"))
    plt.close()


def training(model_name, config, train_input_path, train_label_path, val_input_path, val_label_path, num_train_epochs,
             weight_decay, learning_rate, scheduler_step, check_every, out_path,
             tb_save_path):
    print("start training ")
    train_set, train_regi, train_posi = get_dataset(train_input_path)
    train_iter = DataLoader(train_set, batch_size=config.batch_size, drop_last=True, shuffle=True)
    valid_set, valid_regi, valid_posi = get_dataset(val_input_path)
    val_iter = DataLoader(valid_set, batch_size=config.batch_size, drop_last=True, shuffle=True)

    config.train_regi = train_regi
    config.train_posi = train_posi
    config.valid_regi = valid_regi
    config.valid_posi = valid_posi

    model = Onion(config)
    model.to(device)
    out_path = os.path.join(out_path, model_name + time.strftime("%Y-%m-%d-%X", time.localtime()))
    log_path = os.path.join(out_path, "result.txt")
    tb_save_path = os.path.join(out_path, tb_save_path)

    if not os.path.exists(out_path):
        os.mkdir(out_path)
    if not os.path.exists(tb_save_path):
        os.mkdir(tb_save_path)

    train_model(model, model_name, num_train_epochs, weight_decay, learning_rate, scheduler_step, train_iter, val_iter,
                check_every,
                out_path, tb_save_path, log_path, config)


def train_model(model, model_name, num_train_epochs, weight_decay, learning_rate, scheduler_step, train_iter, val_iter,
                check_every,
                out_path, tb_save_path, log_path, config):
    loss_mse = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_train_epochs,
                                                           eta_min=0, last_epoch=-1, verbose=False)
    writer = SummaryWriter(tb_save_path)

    global_step = 0
    Epoch = tqdm(range(num_train_epochs), desc="Epoch")
    train_loss = 0.0
    train_loss_list = []
    val_loss_list = []
    step = 0
    model.zero_grad()
    for epoch in Epoch:
        # 训练时用train_位置信息
        config.regi = config.train_regi
        config.posi = config.train_posi
        train_loss_epoch = []
        model.train()
        epoch_iterator = tqdm(train_iter, desc="Iteration", mininterval=60)
        for batch_id, (data, label) in enumerate(epoch_iterator):
            data, label = data.to(device), label.to(device)
            output = model(data).to(torch.float32)
            loss = loss_mse(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_value = loss.item()
            train_loss += loss_value
            train_loss_list.append((step, loss_value))
            train_loss_epoch.append(loss_value)
            writer.add_scalar("train_loss", loss_value, step)
            step += 1
        scheduler.step()
        epoch_loss = sum(train_loss_epoch) / len(train_loss_epoch)
        print("\ntrain_loss------>:", epoch_loss)
        with open(log_path, "a") as f:
            f.write("\n epoch:{}, train_loss:{:.10f}".format(epoch, epoch_loss))

            # model_params_path = os.path.join(out_path, model_name + "_params" + str(epoch) + ".pth")
            model_path = os.path.join(out_path, model_name + "_" + str(epoch) + ".pth")
            # torch.save(model.state_dict(), model_params_path)
            torch.save(model, model_path)

        if epoch % 1 == 0:
            # 测试时用Valid_位置信息
            config.regi = config.valid_regi
            config.posi = config.valid_posi
            val_loss = evaluate(model, model_name, val_iter, config.regi, config.posi, loss_mse)
            val_loss_list.append((step - 1, val_loss))
            writer.add_scalar("val_loss", val_loss, epoch, double_precision=True)
            with open(log_path, "a") as f:
                f.write("\n epoch:{}, val_loss:{:.10f}".format(epoch, val_loss))

    loss_str = ["step {}: {}".format(*el) for el in train_loss_list]
    with open(os.path.join(out_path, "train_loss.txt"), "w") as fw:
        fw.write("\n".join(loss_str))

    draw_loss(train_loss_list, val_loss_list, out_path)


def evaluate(model, model_name, val_iter, regi, posi, loss_mse):
    print('\nevaluating')
    ### 交叉验证
    val_loss = 0.0
    val_loss_list = []
    model.eval()
    with torch.no_grad():
        for batch_id, (data, label) in enumerate(val_iter):
            data, label = data.to(device), label.to(device)
            output = model(data)
            loss = loss_mse(output, label)
            val_loss += loss.to(device).item()
            val_loss_list.append(val_loss)
        val_loss = sum(val_loss_list) / len(val_loss_list)
        print("val_loss----->:", val_loss)
    model.train()

    return val_loss


if __name__ == "__main__":
    paser = argparse.ArgumentParser()
    # paser.add_argument("--model_name", help="选择模型", default="expert_mmoe")
    paser.add_argument("--model_name", help="选择模型", default="Onion")
    paser.add_argument("--train_input_path", help="训练集输入数据路径", default="./data_Phantom/phantomdata/mini_train_database.h5")
    paser.add_argument("--train_label_path", help="训练集标签路径",
                       default="./data_HL_2A/train_outputs_0604.h5")  # 没有用，懒得删
    paser.add_argument("--val_input_path", help="验证集输入数据路径", default="./data_Phantom/phantomdata/mini_valid_database.h5")
    paser.add_argument("--val_label_path", help="验证集标签路径", default="./data_HL_2A/val_outputs_0604.h5")  # 没有用，懒得删
    paser.add_argument("--num_train_epochs", help="num_train_epochs", type=int, default=5)
    # paser.add_argument("--batch_size", help="batch_size", type=int, default=4)
    paser.add_argument("--weight_decay", help="weight_decay", type=float, default=0.005)
    paser.add_argument("--learning_rate", help="learning_rate", type=float, default=7e-5)
    paser.add_argument("--scheduler_step", help="lr更新步长", type=int, default=500)
    paser.add_argument("--check_every", help="每多少步validate一次", type=int, default=200)
    paser.add_argument("--out_path", help="输出路径", default="./model_data")
    paser.add_argument("--tb_save_path", help="TensorBoard 保存路径", default="TensorBoard_logs")
    args = paser.parse_args()
    # config = Config(2048, 40, 3, 12, 36 * 32, 0.0, True, torch.float64, 128)
    config = Config(1024, 40, 6, 10, 2500, 1, 0.0, True, torch.float64, 3, 1, 1)
    # file_list = []
    # reg_embd_all = []
    # pos_embd_all = []

    # 遍历当前目录及其子目录
    # for root, dirs, files in os.walk('./data_HL_2A'):  # '.' 表示当前目录
    #     for file in files:
    #         if 'cMatrix' in file:  # 检查文件名是否包含"cregion"
    #             file_list.append(os.path.join(root, file))
    # for i in range(len(file_list)):
    #     aa = np.loadtxt('data_HL_2A/' + str(i) + '_region_list.txt')
    #     bb = np.loadtxt('data_HL_2A/' + str(i) + '_cMatrix.txt')
    #     reg_embd = region_embedding_func(aa, config)
    #     pos_embd = position_embedding_func(bb, config)
    #     reg_embd_all.append(reg_embd.tolist())
    #     pos_embd_all.append(pos_embd.flatten().tolist())

    # config.reg_embd = reg_embd_all
    # config.pos_embd = pos_embd_all

    st = time.time()
    training(model_name=args.model_name,
             config=config,
             train_input_path=args.train_input_path,
             train_label_path=args.train_label_path,
             val_input_path=args.val_input_path,
             val_label_path=args.val_label_path,
             num_train_epochs=args.num_train_epochs,
             weight_decay=args.weight_decay,
             learning_rate=args.learning_rate,
             scheduler_step=args.scheduler_step,
             check_every=args.check_every,
             out_path=args.out_path,
             tb_save_path=args.tb_save_path)
    te = time.time()
    print("time: ", te - st)
