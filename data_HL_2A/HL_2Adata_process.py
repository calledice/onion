import numpy as np
import pandas as pd
import h5py
import os

from tqdm import tqdm

# 获取当前源程序所在的目录
script_dir = os.path.dirname(os.path.abspath(__file__))

# 切换工作目录到源程序所在的目录
os.chdir(script_dir)

# def csv_to_h5(data_root, name):
#     inp_pth = data_root + '/' + name + '_inputs_0604.csv'
#     out_pth = data_root + '/' + name + '_outputs_0604.csv'
#     inp_pth_ = data_root + '/' + name + '_inputs_0604.h5'
#     out_pth_ = data_root + '/' + name + '_outputs_0604.h5'
#
#     df_inp = pd.read_csv(inp_pth, header=None)
#     df_inp[40] = pd.Series([0] * len(df_inp))
#     df_inp[41] = pd.Series([36] * len(df_inp))
#     df_inp[42] = pd.Series([32] * len(df_inp))  # 加入embedding标记
#     df_out = pd.read_csv(out_pth, header=None)
#
#     df_inp.to_hdf(path_or_buf=inp_pth_, key='table')
#     df_out.to_hdf(path_or_buf=out_pth_, key='table')
def extend_row(row,values_to_add):

    return np.concatenate((row, values_to_add))

if __name__ == '__main__':
    # data_root = './data_HL_2A'
    data_root = '../../onion_data/origin_data/data_HL_2A/data'
    name_list = ["val","train","test"]
    if os.path.exists("./data/HL_2A_train_database.h5"):
        print("train set exists.")
        exit(1)       

    os.makedirs("./data",exist_ok=True)

    for name in name_list:
        print(name + ", reading files...")
        # csv_to_h5(data_root,name)
        inp_pth = data_root + '/' + name + '_inputs_0604.csv'
        print(inp_pth + " loaded successfully.")
        out_pth = data_root + '/' + name + '_outputs_0604.csv'
        df_inp = pd.read_csv(inp_pth, header=None)
        df_out = pd.read_csv(out_pth, header=None)
        c_matrix = np.loadtxt(data_root + '/' +'0_cMatrix.txt')/1000
        region = np.loadtxt(data_root + '/' +'0_region_list.txt')
        values_to_add = [0, 32, 36]
        # 删除第一行，这里使用.iloc来排除第一行
        df_inp_dropped_both = df_inp.iloc[1:]
        # 将处理后的DataFrame转换为NumPy数组
        df_inp_array = np.array(df_inp_dropped_both)
        df_out_temp = np.array(df_out.drop(0))
        with open("info.txt", "a") as info:
            info.write(f"samples number = {len(df_inp_array)}")
        with h5py.File("./data"+"/HL_2A_" + name + "_database.h5", 'a') as f:
            input_group = f.create_group("x")
            label_group = f.create_group("y")
            posi_group = f.create_group("posi")
            region_group = f.create_group("regi")

            posi_group.create_dataset(str(0), data=c_matrix)
            region_group.create_dataset(str(0), data=region)

            for i in tqdm(range(len(df_inp_array))):
                df_inp_i = np.append(df_inp_array[i],values_to_add)
                input_group.create_dataset(str(i), data=df_inp_i)
                label_group.create_dataset(str(i), data=df_out_temp[i].reshape(36, 32).T.flatten()*1000)
    print('finish')
