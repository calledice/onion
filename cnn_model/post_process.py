import os

import numpy as np
import matplotlib.pyplot as plt
import json


def plot_data(data, title, save_path,i):
    # 计算最大值和最小值
    max_val = data.max()
    min_val = data.min()

    # 创建等高线级别
    levels = np.linspace(min_val, max_val, 20)
    plt.figure()
    if title != "error":
        plt.pcolor(data, cmap='jet',vmin=0.0,vmax=1.0)
        plt.colorbar(label='ne')
    else:
        plt.pcolor(data, cmap='jet')
        plt.colorbar(label='error(%)')
    plt.title(title)
    ax = plt.gca()
    ax.set_aspect(1.0)
    plt.savefig(save_path + "/" +f"{i}-"+ title)
    plt.show()
    plt.close()
def plot_scatter(input,result,title,save_path,i):
    plt.figure()
    plt.figure(figsize=(5, 4))
    plt.scatter(range(len(input)), input, color='b', marker='^', s=15, alpha=0.8)
    plt.scatter(range(len(result)), result, color='r', marker='o', s=15, alpha=0.5)
    plt.legend(labels=['Input', 'LineIntegral'])
    plt.title(title)
    plt.savefig(save_path + "/" +f"{i}-"+ title)
    # plt.show()
    plt.close()

if __name__ == "__main__":
    pred_path = "./output/Phantom/test/preds.json"
    label_path = "./output/Phantom/test/labels.json"
    input_path = "./output/Phantom/test/inputs.json"
    result_path = "./output/Phantom/test/results.json"

    preds = json.load(open(pred_path, 'r'))
    labels = json.load(open(label_path, 'r'))
    inputs = json.load(open(input_path, 'r'))
    results = json.load(open(result_path, 'r'))
    title_pred = 'preds'
    title_label = 'labels'
    title_error = 'error'
    title_data = "data"
    save_path = "./output/Phantom/figures"
    os.makedirs(save_path,exist_ok=True)
    ave_error_list = []
    error_record_path = save_path + "/error_record.txt"
    for i in range(len(preds)):
        relative_error = abs(np.matrix(preds[i]).T-np.matrix(labels[i]).T)/np.max(np.matrix(labels[i]).T)*100
        ave_error_list.append(np.average(relative_error))
        plot_data(np.matrix(preds[i]).T,title_pred,save_path,i)
        plot_data(np.matrix(labels[i]).T,title_label,save_path,i)
        plot_data(relative_error,title_error,save_path,i)
        plot_scatter(inputs[i],results[i],title_data,save_path,i)
    ave_error_all = np.average(ave_error_list)
    error_all = np.sum(ave_error_list)

    with open(error_record_path,"a") as file:
        file.write(f"ave_error_all = {ave_error_all}\n")
        file.write(f"error_all = {error_all}")
    print(f"ave_error_all = {ave_error_all}")
    print(f"error_all = {error_all}")
    print("good")