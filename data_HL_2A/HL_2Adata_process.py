import numpy as np
import pandas as pd
import h5py
import os

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
    data_root = '/media/congwang/data/python_code/Onion_past/data_HL_2A/data'
    name_list = ["train","test","val"]
    for name in name_list:
        print(name)
        # csv_to_h5(data_root,name)
        inp_pth = data_root + '/' + name + '_inputs_0604.csv'
        out_pth = data_root + '/' + name + '_outputs_0604.csv'
        df_inp = pd.read_csv(inp_pth, header=None)
        df_out = pd.read_csv(out_pth, header=None)
        c_matrix = np.loadtxt(data_root + '/' +'0_cMatrix.txt')
        region = np.loadtxt(data_root + '/' +'0_region_list.txt')
        with h5py.File("./data"+"/" + name + "_database_test.h5", 'a') as f:
            input_group = f.create_group("x")
            label_group = f.create_group("y")
            posi_group = f.create_group("posi")
            region_group = f.create_group("regi")

            posi_group.create_dataset(str(0), data=c_matrix)
            region_group.create_dataset(str(0), data=region)
            values_to_add = [0, 36, 32]
            # 删除第一列 不需要
            # df_inp_dropped_col = df_inp.drop(df_inp.columns[0], axis=1)
            # 删除第一行，这里使用.iloc来排除第一行
            df_inp_dropped_both = df_inp.iloc[1:]
            # 将处理后的DataFrame转换为NumPy数组
            df_inp_array = np.array(df_inp_dropped_both)
            df_out_array = np.array(df_out.drop(0))
            for i in range(len(df_inp_array)):
                df_inp_i = np.append(df_inp_array[i],values_to_add)
                input_group.create_dataset(str(i), data=df_inp_i)
                label_group.create_dataset(str(i), data=df_out_array[i])
    print('finish')
