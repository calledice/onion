import os
import numpy as np
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
from pathlib import Path
import re

# 获取当前源程序所在的目录
script_dir = os.path.dirname(os.path.abspath(__file__))

# 切换工作目录到源程序所在的目录
os.chdir(script_dir)

def plot_data(data,title, save_path,i,max_val):
    # 计算最大值和最小值
    max_val = max_val
    min_val = 0
    # 创建等高线级别
    if title != 'Relative error':
        plt.figure()
        cs = plt.pcolor(data, cmap='jet',vmin=0.0,vmax=1.0*max_val)
        cbar = plt.colorbar(cs)
        cbar.set_label(r'$I_{SX} (a.u.)$', fontsize=16)
        cbar.ax.tick_params(labelsize=16) 
        plt.xlabel('$\t{R}_{index}$',fontsize=16)
        plt.ylabel('$\t{Z}_{index}$',fontsize=16)
        plt.title(title,fontsize=16)
        ax = plt.gca()
        ax.set_aspect(1.0)
        ax.tick_params(axis='both', which='major', labelsize=16)
        plt.savefig(save_path + "/" +f"{i}-"+ title,dpi = 300,bbox_inches='tight')
    else:
        plt.figure()
        # cs = plt.pcolor(data, cmap='jet')
        cs1 = plt.pcolor(data, cmap='jet',vmin=0.0,vmax=0.035)
        cbar = plt.colorbar(cs1)
        cbar.set_label(r'$\epsilon_1$', fontsize=16)
        cbar.ax.tick_params(labelsize=16) 
        plt.xlabel('$\t{R}_{index}$',fontsize=16)
        plt.ylabel('$\t{Z}_{index}$',fontsize=16)
        plt.title(title,fontsize=16)
        ax = plt.gca()
        ax.set_aspect(1.0)
        ax.tick_params(axis='both', which='major', labelsize=16)
        plt.savefig(save_path + "/" +f"{i}-"+ "error",dpi = 300, bbox_inches='tight')
    # plt.show()
    plt.close()
def plot_scatter(input,result,label2results,ave_label2result_error,title,save_path,i):
    plt.figure()
    error = np.array(ave_label2result_error) * abs(np.max(input))
    plt.scatter(range(len(input)), input, color='b', marker='^', s=15, alpha=0.8)
    # plt.scatter(range(len(result)), result, color='g', marker='o', s=15, alpha=0.5)
    plt.scatter(range(len(label2results)), label2results, color='r', marker='o', s=15, alpha=0.5)
    # plt.errorbar(range(len(input)), input, yerr=error, fmt='none', ecolor='r', capsize=5, alpha=0.5)
    plt.legend(labels=['SXR data','Target profile@BP'])
    # plt.legend(labels=['SXR data',"Reconstuction profile@BP",'Target profile@BP'])
    plt.xlabel('$\t{Channel}$',fontsize=16)
    plt.ylabel('$\t{I}_{SXR}(a.u.)$',fontsize=16)
    plt.title(title,fontsize=16)
    ax = plt.gca()
    # ax.set_aspect(1.0)
    ax.tick_params(axis='both', which='major', labelsize=16)
    plt.savefig(save_path + "/" +f"{i}-"+ title,dpi = 300, bbox_inches='tight')
    # plt.show()
    plt.close()
def plot_loss(train_losses, val_losses, out_dir):
    import matplotlib.pyplot as plt
    iters = list(range(len(train_losses)))
    # 创建一个新的图形
    plt.figure()
    # 绘制第一条曲线
    plt.plot(iters, train_losses, label='Training loss', color='blue')
    # 绘制第二条曲线
    plt.plot(iters, val_losses, label='Validation loss', color='red')
    # 添加标题和标签
    plt.legend(fontsize=16)
    plt.title('Loss curve',fontsize=16)
    plt.xlabel('Epochs',fontsize=16)
    plt.ylabel('Loss',fontsize=16)
    # 显示图例
    ax = plt.gca()
    # ax.set_aspect(1.0)
    ax.tick_params(axis='both', which='major', labelsize=16)
    plt.savefig(f'{out_dir}/loss_curve_new.png',dpi=300, bbox_inches='tight')
    # 显示图形
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
        file.write(f"ave_label2result_error_all = {ave_label2result_error_all}\n")
        file.write(f"ave_pre2result_error_all = {ave_pre2result_error_all}\n")
        # file.write(f"error_all = {error_all}")
    print(f"ave_error_all = {ave_error_all}")
    print(f"ave_label2result_error_all = {ave_label2result_error_all}")
    print(f"ave_pre2result_error_all = {ave_pre2result_error_all}")
    # print(f"error_all = {error_all}")
    print("good")

def visualize_up(preds,labels,inputs,results,label2results,case_file):
    print("start visualize")
    title_pred = 'Reconstuction profile'
    title_label = 'Target profile'
    title_error = 'Relative error'
    title_data = "SXR data and BPs"
    save_path = case_file+"/figures_new_0.035"
    os.makedirs(save_path,exist_ok=True)
    ave_error_list = []
    ave_label2result_error2_list = []# 由label获得的弦积分结果与input的偏差
    ave_pre2result_error2_list = []# 由pre获得的弦积分结果与input的偏差
    error_record_path = case_file+"/error_record.txt"
    for i in tqdm(range(len(preds)), desc='Visualizing'):
        relative_error1 = abs(np.matrix(preds[i])-np.matrix(labels[i]))/np.max(np.matrix(labels[i]))
        ave_label2result_error2 = abs(np.array(label2results[i])-np.array(inputs[i]))/np.max(np.array(inputs[i]))# E2
        ave_pre2result_error2 = abs(np.array(results[i])-np.array(inputs[i]))/np.max(np.array(inputs[i]))# E2
        # ave_label2result_error2 = abs(np.array(label2results[i])-np.array(inputs[i]))/np.array(inputs[i])# E2-up
        # ave_pre2result_error2 = abs(np.array(results[i])-np.array(inputs[i]))/np.array(inputs[i])# E2-up
        ave_error_list.append(np.average(relative_error1))
        ave_label2result_error2_list.append(np.average(ave_label2result_error2))
        ave_pre2result_error2_list.append(np.average(ave_pre2result_error2))
        if i < 10:
            max_val = max(np.max(preds[i]),np.max(labels[i]))
            plot_data(np.matrix(preds[i]),title_pred,save_path,i,max_val)
            plot_data(np.matrix(labels[i]),title_label,save_path,i,max_val)
            max_val_error = np.max(relative_error1)
            plot_data(relative_error1,title_error,save_path,i,max_val_error)
            plot_scatter(inputs[i],results[i],label2results[i],ave_label2result_error2,title_data,save_path,i)
    ave_error_all = np.average(ave_error_list)
    ave_label2result_error_all = np.average(ave_label2result_error2_list)
    ave_pre2result_error_all = np.average(ave_pre2result_error2_list)
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

def dataset_evaluation(inputs,label2results,case_file):
    print("start dataset_evaluation")
    save_path = case_file
    os.makedirs(save_path,exist_ok=True)
    ave_error_list = []
    ave_label2result_error2_list = []# 由label获得的弦积分结果与input的偏差
    error_record_path = case_file+"/error_record.txt"
    for i in tqdm(range(len(inputs)), desc='Visualizing'):
        ave_label2result_error2 = abs(np.array(label2results[i])-np.array(inputs[i]))/np.max(np.array(inputs[i]))# E2
        # ave_label2result_error2 = abs(np.array(label2results[i])-np.array(inputs[i]))/np.array(inputs[i])# E2-up
        ave_label2result_error2_list.append(np.average(ave_label2result_error2))
    ave_error_all = np.average(ave_error_list)
    ave_label2result_error_all = np.average(ave_label2result_error2_list)
    # error_all = np.sum(ave_error_list)

    with open(error_record_path,"a") as file:
        file.write(f"ave_error_all = {ave_error_all}\n")
        file.write(f"ave_label2result_all = {ave_label2result_error_all}\n")
        # file.write(f"error_all = {error_all}")
    print(f"ave_error_all = {ave_error_all}")
    print(f"ave_label2result_all = {ave_label2result_error_all}")
    # print(f"error_all = {error_all}")
    print("good")


def find_json_files(directory, prefix):
    path = Path(directory)
    pattern = f"{prefix}*.json"
    return sorted(
        path.glob(pattern), 
        key=lambda p: int(re.search(rf'{prefix}(\d+)', p.name).group(1)) if re.search(rf'{prefix}(\d+)', p.name) else -1
    )
def parse_json_file(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)  # Assuming the JSON file contains a list or array of numbers
    return data
if __name__ == "__main__": #重新画图
    
    case_path = "../../onion_train_data/train_results_2A/"
    # # 批量重新画图
    # path = Path(case_path)
    # # 获取该层级的所有文件夹，不会递归到子文件夹中
    # folders = [f.name for f in path.iterdir() if f.is_dir()]
    # for name in folders:
    #     case_file = case_path + name
    #     pred_path = case_file+"/test/preds.json"
    #     label_path = case_file+"/test/labels.json"
    #     input_path = case_file+"/test/inputs.json"
    #     result_path = case_file+"/test/results.json"# pre2result
    #     label2result_path = case_file+"/test/label2results.json"# label2results

    #     preds = json.load(open(pred_path, 'r'))
    #     labels = json.load(open(label_path, 'r'))
    #     inputs = json.load(open(input_path, 'r'))
    #     results = json.load(open(result_path, 'r'))
    #     label2results = json.load(open(label2result_path, 'r'))
    #     print("Files loaded successfully.")
    #     visualize_up(preds,labels,inputs,results,label2results,case_file)
    # 单个重新画图
    name = "EXP2A_42_ResOnion_input__softplus"
    print(f"start {name}")
    case_file = case_path + name
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
    visualize_up(preds,labels,inputs,results,label2results,case_file)
    


# if __name__ == "__main__": # 可视化loss
#     print("start")
#     case_path = "../../onion_train_data/train_results_EAST/"
#     # case_path = "../../onion_train_data/train_results_2A/"
#     # 创建Path对象
#     path = Path(case_path)
#     # 获取该层级的所有文件夹，不会递归到子文件夹中
#     folders = [f.name for f in path.iterdir() if f.is_dir()]
#     for name in folders:
#         train_losses = []
#         val_losses = []
#         case_file = case_path + name
#         train_path = case_file+"/train/"
#         prefix_train = 'training_loss'
#         prefix_val = 'val_loss'
#         train_json_files = find_json_files(train_path, prefix_train)
#         val_json_files = find_json_files(train_path, prefix_val)
#         for train_json_file in train_json_files:
#             train_loss = parse_json_file(train_json_file)
#             train_loss_ave = np.average(train_loss)
#             train_losses.append(train_loss_ave)
#         for val_json_file in val_json_files:
#             val_loss = parse_json_file(val_json_file)
#             val_loss_ave = np.average(val_loss)
#             val_losses.append(val_loss_ave)
#         plot_loss(train_losses, val_losses, train_path)
#         print("good")

# if __name__ == "__main__": #评估数据集
#     print("start")
#     case_path = "../../onion_train_data/train_results_EAST/"
#     # 创建Path对象
#     path = Path(case_path)
#     # 获取该层级的所有文件夹，不会递归到子文件夹中
#     folders = [f.name for f in path.iterdir() if f.is_dir()]
#     for name in folders:
#         case_file = case_path + name
#         pred_path = case_file+"/test/preds.json"
#         label_path = case_file+"/test/labels.json"
#         input_path = case_file+"/test/inputs.json"
#         result_path = case_file+"/test/results.json"# pre2result
#         label2result_path = case_file+"/test/label2results.json"# label2results

#         preds = json.load(open(pred_path, 'r'))
#         labels = json.load(open(label_path, 'r'))
#         inputs = json.load(open(input_path, 'r'))
#         results = json.load(open(result_path, 'r'))
#         label2results = json.load(open(label2result_path, 'r'))
#         print("Files loaded successfully.")
#         visualize_up(preds,labels,inputs,results,label2results,case_file)