import numpy as np
import scipy.io as sio
import fnmatch
import os
import glob
import h5py
# def find_mat_files(directory):
#     # 递归地查找所有.mat文件
#     return [os.path.join(root, name)
#             for root, dirs, files in os.walk(directory)
#             for name in files
#             if name.endswith('.mat')]

def search_mat_files(directory):
    """搜索目录及其所有子目录中的 .mat 文件"""
    mat_files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, '*.mat'):
            mat_files.append(os.path.join(root, filename))
    return mat_files

# 使用方法
directory_path = '/data_East/data/DATA_set_MCW'
mat_files_list = search_mat_files(directory_path)
c_matrix_sss = sio.loadmat("/data_East/data/Poly_SXR.mat")
c_matrix_s = c_matrix_sss['Poly_SXR'][0][0]
c_matrix_ss = c_matrix_sss['Poly_SXR'][1][0]
c_matrix_1 = c_matrix_s.reshape(-1, c_matrix_s.shape[2]).T
c_matrix_2 = c_matrix_ss.reshape(-1, c_matrix_ss.shape[2]).T
c_matrix = np.vstack((c_matrix_1, c_matrix_2))
values_to_add = [0, 75, 50]
lengh_vec= values_to_add[1]*values_to_add[2]
region =  np.zeros(lengh_vec)

with h5py.File("/media/congwang/data/python_code/Onion/data_East/DATA_set_MCW/EAST_database.h5", 'a') as f:
    input_group = f.create_group("x")
    label_group = f.create_group("y")
    posi_group = f.create_group("posi")
    region_group = f.create_group("regi")

    posi_group.create_dataset(str(0), data=c_matrix)#40*1152
    region_group.create_dataset(str(0), data=region)#1152
    # 调用函数并处理每个找到的.mat文件
    j = 0
    for file_path in mat_files_list:
        print(file_path)
        data = sio.loadmat(file_path)
        xu_mean = np.mean(data["Xu"], axis=1)  # 注意axis参数
        xd_mean = np.mean(data["Xd"], axis=1)  # 注意axis参数
        x = np.concatenate((xu_mean,xd_mean))
        y = data["Sm"].reshape(lengh_vec,1)#1152*1
        df_inp_array = np.array(x)
        df_out_array = np.array(y)
        df_inp_i = np.append(df_inp_array, values_to_add)
        input_group.create_dataset(str(j), data=df_inp_i)
        label_group.create_dataset(str(j), data=df_out_array.T[0])
        j+=1

print("finish")