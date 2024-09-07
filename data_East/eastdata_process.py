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
directory_path = 'D:\Onion_data\data_East\data\DATA_set_MCW'
mat_files_list = search_mat_files(directory_path)
c_matrix_sss = sio.loadmat("./Poly_SXR.mat")
c_matrix_s = c_matrix_sss['Poly_SXR'][0][0]
c_matrix_ss = c_matrix_sss['Poly_SXR'][1][0]
c_matrix_1 = c_matrix_s.reshape(-1, c_matrix_s.shape[2]).T
c_matrix_2 = c_matrix_ss.reshape(-1, c_matrix_ss.shape[2]).T
c_matrix = np.vstack((c_matrix_1, c_matrix_2))*1000
values_to_add = [0, 75, 50]
lengh_vec= values_to_add[1]*values_to_add[2]
region =  np.ones(lengh_vec)
j = 0
input_list = []
label_list = []
for file_path in mat_files_list:
    print(file_path)
    data = sio.loadmat(file_path)
    xu_mean = np.mean(data["Xu"], axis=1)  # 注意axis参数
    xd_mean = np.mean(data["Xd"], axis=1)  # 注意axis参数
    x = np.concatenate((xu_mean,xd_mean))
    y = data["Sm"].reshape(lengh_vec,1)#3750*1
    df_inp_array = np.array(x)
    df_out_array = np.array(y)
    df_inp_i = np.append(df_inp_array, values_to_add)
    input_list.append(df_inp_i)
    label_list.append(df_out_array.T[0])

assert len(input_list) == len(label_list), "输入列表和标签列表长度必须相同"
print("load finish")
# 合并输入和标签
combined = list(zip(input_list, label_list))
# 设置随机种子以保证实验的可重复性
np.random.seed(42)
# 打乱合并后的列表
np.random.shuffle(combined)
# 解压回原来的列表
shuffled_inputs, shuffled_labels = zip(*combined)
# 转换成numpy数组以便更容易处理
shuffled_inputs = list(shuffled_inputs)
shuffled_labels = list(shuffled_labels)
# 数据集大小
data_size = len(shuffled_inputs)
# 划分比例
train_split = 0.7
val_split = 0.2
test_split = 0.1
# 计算分割索引
train_end = int(train_split * data_size)
val_end = int((train_split + val_split) * data_size)
# 分割数据集
train_inputs, train_labels = shuffled_inputs[:train_end], shuffled_labels[:train_end]
val_inputs, val_labels = shuffled_inputs[train_end:val_end], shuffled_labels[train_end:val_end]
test_inputs, test_labels = shuffled_inputs[val_end:], shuffled_labels[val_end:]
name = ['train','valid','test']
os.makedirs("./data",exist_ok=True)

with h5py.File(f"./data/EAST_{name[0]}_database.h5", 'a') as data0:
    data_input_group = data0.create_group("x")
    data_label_group = data0.create_group("y")
    data_posi_group = data0.create_group("posi")
    data_region_group = data0.create_group("regi")
    for j in range(len(train_inputs)):
        data_input_group.create_dataset(str(j), data=train_inputs[j])
        data_label_group.create_dataset(str(j), data=train_labels[j])
    data_posi_group.create_dataset(str(0), data=c_matrix)
    data_region_group.create_dataset(str(0), data=region)

with h5py.File(f"./data/EAST_{name[1]}_database.h5", 'a') as data1:
    data_input_group = data1.create_group("x")
    data_label_group = data1.create_group("y")
    data_posi_group = data1.create_group("posi")
    data_region_group = data1.create_group("regi")
    for j in range(len(val_inputs)):
        data_input_group.create_dataset(str(j), data=val_inputs[j])
        data_label_group.create_dataset(str(j), data=val_labels[j])
    data_posi_group.create_dataset(str(0), data=c_matrix)
    data_region_group.create_dataset(str(0), data=region)

with h5py.File(f"./data/EAST_{name[2]}_database.h5", 'a') as data2:
    data_input_group = data2.create_group("x")
    data_label_group = data2.create_group("y")
    data_posi_group = data2.create_group("posi")
    data_region_group = data2.create_group("regi")
    for j in range(len(test_inputs)):
        data_input_group.create_dataset(str(j), data=test_inputs[j])
        data_label_group.create_dataset(str(j), data=test_labels[j])
    data_posi_group.create_dataset(str(0), data=c_matrix)#92*（75*50）
    data_region_group.create_dataset(str(0), data=region)#75*50

print("finish")