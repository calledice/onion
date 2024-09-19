import os
import numpy as np
import matplotlib.pyplot as plt
import json
from tqdm import tqdm

# 获取当前源程序所在的目录
script_dir = os.path.dirname(os.path.abspath(__file__))

# 切换工作目录到源程序所在的目录
os.chdir(script_dir)

def plot_data(data, title, save_path,i,max_val):
    # 计算最大值和最小值
    max_val = max_val
    min_val = 0

    # 创建等高线级别
    levels = np.linspace(min_val, max_val, 20)
    plt.figure()
    if title != "error":
        plt.pcolor(data, cmap='jet',vmin=0.0,vmax=1.0*max_val)
        plt.colorbar(label='ne')
    else:
        plt.pcolor(data, cmap='jet')
        plt.colorbar(label='error')
    plt.title(title)
    ax = plt.gca()
    ax.set_aspect(1.0)
    plt.savefig(save_path + "/" +f"{i}-"+ title,dpi = 300)
    # plt.show()
    plt.close()
def plot_scatter(input,result,label2results,ave_label2result_error,title,save_path,i):
    plt.figure()
    plt.figure(figsize=(5, 4))
    error = np.array(ave_label2result_error) * abs(np.max(input))
    plt.scatter(range(len(input)), input, color='b', marker='^', s=15, alpha=0.8)
    plt.scatter(range(len(result)), result, color='g', marker='o', s=15, alpha=0.5)
    plt.scatter(range(len(label2results)), label2results, color='r', marker='o', s=15, alpha=0.5)
    plt.errorbar(range(len(label2results)), input, yerr=error, fmt='none', ecolor='r', capsize=5, alpha=0.5)
    # plt.legend(labels=['Input','Label@C_matrix'])
    plt.legend(labels=['Input','Label@C_matrix',"Pre@C_matrix"])
    plt.title(title)
    plt.savefig(save_path + "/" +f"{i}-"+ title,dpi = 300)
    # plt.show()
    plt.close()

def visualize(case_file):
    # case_file = "Phantom_Onion_gavin"
    pred_path = case_file+"/test/preds.json"
    label_path = case_file+"/test/labels.json"
    input_path = case_file+"/test/inputs.json"
    result_path = case_file+"/test/results.json"# pre2result
    label2result_path = case_file+"/test/label2results.json"# label2results

    preds = json.load(open(pred_path, 'r'))
    labels = json.load(open(label_path, 'r'))
    inputs = json.load(open(input_path, 'r'))
    results = json.load(open(result_path, 'r'))
    label2results = json.load(open(label2result_path, 'r'))
    print("Files loaded successfully.")
    title_pred = 'preds'
    title_label = 'labels'
    title_error = 'error'
    title_data = "data"
    save_path = case_file+"/figures"
    os.makedirs(save_path,exist_ok=True)
    ave_error_list = []
    ave_label2result_error_list = []# 由label获得的弦积分结果与input的偏差
    ave_pre2result_error_list = []# 由pre获得的弦积分结果与input的偏差
    error_record_path = case_file+"/error_record.txt"
    for i in tqdm(range(len(preds)), desc='Visualizing'):
        relative_error = abs(np.matrix(preds[i])-np.matrix(labels[i]))/np.max(np.matrix(labels[i]))
        ave_label2result_error = abs(np.array(label2results[i])-np.array(inputs[i]))/np.max(np.array(inputs[i]))
        ave_pre2result_error = abs(np.array(results[i])-np.array(inputs[i]))/np.max(np.array(inputs[i]))
        ave_error_list.append(np.average(relative_error))
        ave_label2result_error_list.append(np.average(ave_label2result_error))
        ave_pre2result_error_list.append(np.average(ave_pre2result_error))
        if i < 10:
            max_val = max(np.max(preds[i]),np.max(labels[i]))
            plot_data(np.matrix(preds[i]),title_pred,save_path,i,max_val)
            plot_data(np.matrix(labels[i]),title_label,save_path,i,max_val)
            max_val_error = np.max(relative_error)
            plot_data(relative_error,title_error,save_path,i,max_val_error)
            plot_scatter(inputs[i],results[i],label2results[i],ave_label2result_error,title_data,save_path,i)
    ave_error_all = np.average(ave_error_list)
    ave_label2result_error_all = np.average(ave_label2result_error_list)
    ave_pre2result_error_all = np.average(ave_pre2result_error_list)
    # error_all = np.sum(ave_error_list)

    with open(error_record_path,"a") as file:
        file.write(f"ave_error_all = {ave_error_all}\n")
        file.write(f"ave_label2result_all = {ave_label2result_error_all}\n")
        file.write(f"ave_pre2result_error_all = {ave_pre2result_error_all}\n")
        # file.write(f"error_all = {error_all}")
    print(f"ave_error_all = {ave_error_all}")
    print(f"ave_label2result_all = {ave_label2result_error_all}")
    print(f"ave_pre2result_error_all = {ave_pre2result_error_all}")
    # print(f"error_all = {error_all}")
    print("good")

if __name__ == "__main__":
    print("start")
    case_file = "/mnt/d/project/onion_data/model_train/output_v0/phantom2A_42_Onion_PI_addloss_softplus"
    visualize(case_file)
