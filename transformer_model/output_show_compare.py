import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from dataset import OnionDataset
from torch.utils.data import DataLoader
import torch
from onion_model import Onion, Config, ConvEmbModel
import json
import torch.nn as nn
import time
import h5py
import os
import glob

def plot_data(data, title, save_path,i):
    max = data.max()
    min = data.min()
    plt.figure()
    plt.pcolor(data, cmap='jet',vmin=0.0,vmax=1.0)
    plt.colorbar(label='ne')
    plt.title(title)
    ax = plt.gca()
    ax.set_aspect(1.0)
    plt.savefig(save_path + "/" +f"{i}-"+ title)
    # plt.show()
    plt.close()


def scatter_data(gt_data, pre_data, title, save_path,i):
    plt.figure()
    plt.figure(figsize=(5, 4))
    plt.scatter(range(len(gt_data)), gt_data, color='b', marker='^', s=15, alpha=0.8)
    plt.scatter(range(len(gt_data)), pre_data, color='r', marker='o', s=15, alpha=0.5)
    plt.legend(labels=['Input', 'LineIntegral'])
    plt.title(title)
    plt.savefig(save_path + "/" +f"{i}-"+ title)
    # plt.show()
    plt.close()


def predict(config, model_load_path, model_name, test_iter,model_path):
    model = torch.load(model_load_path, map_location=torch.device('cpu'))
    model = model.to(device)
    loss_mse = nn.MSELoss()
    model.eval()
    preds_all, labels_all,info_all,input_all,result_all = [], [],[],[],[]
    test_loss = 0.0
    test_loss_list = []
    t1 = time.time()
    with torch.no_grad():
        for batch_id, ((input, regi, posi, info), label) in enumerate(test_iter):
            input, regi, posi, label = input.to(device), regi.to(device), posi.to(device), label.to(device)
            output = model(input, regi, posi).squeeze(1)
            output_temp = output.unsqueeze(-1)  #
            result = torch.bmm(posi, output_temp).squeeze(-1)  #
            sigmoid_n = torch.sigmoid(model.n)
            loss = loss_mse(output, label) + sigmoid_n * loss_mse(input, result)
            test_loss += loss.to(device).item()
            test_loss_list.append(test_loss)
            preds_all.extend(output.cpu().numpy())
            labels_all.extend(label.cpu().numpy())
            info_all.extend(np.array(info).T)
            input_all.extend(input.cpu().numpy())
            result_all.extend(result.cpu().numpy())
        preds_all = np.array(preds_all)
        labels_all = np.array(labels_all)
        info_all = np.array(info_all)
        input_all = np.array(input_all)
        result_all = np.array(result_all)
        mae = np.mean(np.abs(preds_all - labels_all))
        print(f'test set mae = np.mean(np.abs(preds - labels)): {mae}')
        df_preds = pd.DataFrame(preds_all)
        df_labels = pd.DataFrame(labels_all)
        df_info = pd.DataFrame(info_all)
        df_input = pd.DataFrame(input_all)
        df_result = pd.DataFrame(result_all)

    t2 = time.time()
    avg_test_loss = sum(test_loss_list) / len(test_loss_list)
    print(f'test set loss is: {avg_test_loss}, test time is: {t2 - t1} s')

    # 循环结束后，将所有输出写入csv文件
    output_path = model_path+"/outputs"
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    if not os.path.exists(output_path+"/"+ model_name):
        os.mkdir(output_path +"/" +model_name)

    pre_path = output_path+"/{}/pre_matrix_.csv".format(model_name)
    gt_path = output_path+"/{}/gt_matrix_.csv".format(model_name)
    info_path = output_path+"/{}/info_matrix_.csv".format(model_name)
    input_path = output_path+"/{}/input_matrix_.csv".format(model_name)
    result_path = output_path+"/{}/result_matrix_.csv".format(model_name)
    df_preds.to_csv(pre_path)
    df_labels.to_csv(gt_path)
    df_info.to_csv(info_path)
    df_input.to_csv(input_path)
    df_result.to_csv(result_path)
    print('finish predict and save in output file')
    return pre_path, gt_path, info_path,input_path,result_path

def plot_save(df_pre,df_gt,df_info,df_input,df_result,path):
    num = len(df_info)
    for i in range(10):
        insize = df_info[i][0]
        r = df_info[i][1]
        z = df_info[i][2]
        outsize = r*z
        Pre = df_pre[i][:outsize].reshape(r,z).T
        Label = df_gt[i][:outsize].reshape(r,z).T
        RelativeError = (abs(df_gt[i][:outsize].reshape(r, z)-df_pre[i][:outsize].reshape(r,z))/np.max(df_gt[i][:outsize].reshape(r,z))).T
        scatter_data(df_input[i][:insize],df_result[i][:insize],"LineIntegral&GT",path,i)
        plot_data(Pre,"Pre",path,i)
        plot_data(Label,"Label",path,i)
        plot_data(RelativeError, "RelativeError", path, i)

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #######################################################
    # 查看test/train数据
    # test_input_path = "./data_Phantom/phantomdata/mini_1_test_database.h5"
    # dataset = h5py.File(test_input_path, 'r')
    # x = dataset["x"]
    # y = dataset["y"]
    # regi = dataset['regi']
    # posi = dataset['posi']
    # inputs_list = [x[dkey][:][:-3].flatten() for dkey in x.keys()]  # 收集输入数据
    # outputs_list = [y[dkey][:].flatten() for dkey in y.keys()]
    # plt.contourf(y["0"][:].reshape(int(x["0"][-1]),int(x["0"][-2])), cmap='jet')
    # plt.show()
    # print("good")
    ######################################################
    test = True
    model_name = "Onion_9"
    model_path = "./model_attn_data/Onion2024-08-26-14:15:07/"
    modelPath = model_path + model_name + ".pth"
    # json_file = model_path + "config_and_args_4L_norm_2loss_2048.json"
    json_file = glob.glob(os.path.join(model_path, '*.json'))
    # 加载模型进行test
    if test:
        with open(json_file, 'r', encoding='utf-8') as file:
            data = json.load(file)
        config = Config(data['config']["n_layer"], data['config']["n_head"],data['config']["dropout"],data['config']["bias"],
            data['config']["dtype"],data['config']["batch_size"],data['config']["max_input_len"],data['config']["max_rz_len"])

        test_input_path = "../data_Phantom/phantomdata/mini_1_test_database_1_100_1000.h5"
        test_set = OnionDataset(test_input_path, max_input_len=config.max_input_len, max_rz_len=config.max_rz_len)
        test_iter = DataLoader(test_set, batch_size=config.batch_size, drop_last=True, shuffle=False)
        pre_path,gt_path,info_path,input_path,result_path = predict(config = config, model_load_path=modelPath, model_name=model_name,test_iter=test_iter,model_path = model_path)
    else:
        pre_path = model_path + f"outputs/{model_name}/pre_matrix_.csv"
        gt_path = model_path + f"outputs/{model_name}/gt_matrix_.csv"
        info_path = model_path + f"outputs/{model_name}/info_matrix_.csv"
        input_path = model_path + f"outputs/{model_name}/input_matrix_.csv"
        result_path = model_path + f"outputs/{model_name}/result_matrix_.csv" #是弦积分后的结果

    path = os.path.split(pre_path)[0]
    df_pre = pd.read_csv(pre_path).values[:,1:]
    df_gt = pd.read_csv(gt_path).values[:,1:]
    df_info = pd.read_csv(info_path).values[:,1:]
    df_input = pd.read_csv(input_path).values[:,1:]
    df_result = pd.read_csv(result_path).values[:,1:]
    print("finish read from csv start plot")
    plot_save(df_pre,df_gt,df_info,df_input,df_result,path)
