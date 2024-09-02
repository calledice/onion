import os

import numpy as np
import math
import random
import h5py
import matplotlib.pyplot as plt

class BeamDef:
    def __init__(self, beams):
        num = len(beams)
        self.dr = [0.0] * num
        self.dz = [0.0] * num
        self.r = [0.0] * num
        self.z = [0.0] * num

        for i in range(num):
            beam = beams[i]
            self.r[i] = beam[0]
            self.z[i] = beam[1]
            self.dr[i] = beam[2]
            self.dz[i] = beam[3]
        self.current = None


def find_edgebeams(beams):
    nbeams = len(beams.r)
    idxs = []

    for i in range(nbeams):
        hasup = False
        hasdown = False
        hasleft = False
        hasright = False

        for k in range(nbeams):
            if abs(beams.r[k] - beams.r[i]) < (beams.dr[k] + beams.dr[i]) / 2:
                if beams.z[k] < beams.z[i]:
                    hasup = True
                if beams.z[k] > beams.z[i]:
                    hasdown = True

            if abs(beams.z[k] - beams.z[i]) < (beams.dz[k] + beams.dz[i]) / 2:
                if beams.r[k] < beams.r[i]:
                    hasleft = True
                if beams.r[k] > beams.r[i]:
                    hasright = True
        if not hasup or not hasdown or not hasleft or not hasright:
            idxs.append(i)

    return idxs


################################################################
def grid_generate(minr, maxr, numr, minz, maxz, numz):
    dr = (maxr - minr) / (numr - 1)
    dz = (maxz - minz) / (numz - 1)
    beams = []

    for i in range(numr):
        r = minr + i * dr
        for j in range(numz):
            z = minz + j * dz
            beams.append([r, z, dr, dz])

    return BeamDef(beams)


def get_cmatrix(grid_defr, grid_defz, los_pos, los_angle, grid):
    # 加载网格
    emission_node = grid
    pixel_r = np.array(emission_node.r)
    pixel_z = np.array(emission_node.z)

    grid_r = grid_defr
    grid_z = grid_defz

    deltar_half = (grid_r[1] - grid_r[0]) / 2
    deltaz_half = (grid_z[1] - grid_z[0]) / 2
    # 建立观测弦坐标
    los_startrz = los_pos
    start_r = los_startrz[0]
    start_z = los_startrz[1]

    extend_r = [start_r[i] + math.cos(math.pi * los_angle[i] / 180) * 1e10 for i in range(len(start_r))]
    extend_z = [start_z[i] + math.sin(math.pi * los_angle[i] / 180) * 1e10 for i in range(len(start_z))]

    # 求交点和距离
    c = []
    for s_r, e_r, s_z, e_z in zip(start_r, extend_r, start_z, extend_z):
        # 求观测弦表达式
        l_a = (s_z - e_z) / (s_r - e_r)
        l_b = (e_z - l_a * e_r)
        for px_r, px_z in zip(pixel_r, pixel_z):## r,z
            intersections = []
            intersect_left = (px_r - deltar_half) * l_a + l_b
            intersect_right = (px_r + deltar_half) * l_a + l_b
            intersect_up = ((px_z + deltaz_half) - l_b) / l_a
            intersect_down = ((px_z - deltaz_half) - l_b) / l_a

            if px_z - deltaz_half < intersect_left < px_z + deltaz_half:
                intersections.append([px_r - deltar_half, intersect_left])
            if px_z - deltaz_half < intersect_right < px_z + deltaz_half:
                intersections.append([px_r + deltar_half, intersect_right])
            if px_r - deltar_half < intersect_up < px_r + deltar_half:
                intersections.append([intersect_up, px_z + deltaz_half])
            if px_r - deltar_half < intersect_down < px_r + deltar_half:
                intersections.append([intersect_down, px_z - deltaz_half])

            if len(intersections) == 2:
                c.append(((intersections[0][0] - intersections[1][0]) ** 2 + (
                        intersections[0][1] - intersections[1][1]) ** 2) ** 0.5)
            else:
                c.append(0.)
    M = np.matrix(c).reshape(len(los_angle), -1)
    return M


def region_generate(randn_indexr, randn_indexz, numgridr, numgridz, grid):
    label_raw = np.zeros(len(grid.r)).reshape(numgridr, numgridz)
    label_dis = np.zeros(len(grid.r)).reshape(numgridr, numgridz)
    label_raw[randn_indexr, randn_indexz] = 1
    grad_coff = random.randint(1, 9)
    grad_coff = grad_coff/(-20.0)
    threshhold_coff = random.randint(5, 20)
    threshhold_coff = 1/threshhold_coff
    for row in range(numgridr):
        for col in range(numgridz):
            distnace = (abs(row - randn_indexr)**2 + abs(col - randn_indexz)**2)**0.5/3
            label_dis[row, col] = distnace
    for row in range(numgridr):
        for col in range(numgridz):
            label_raw[row, col] = 1 * np.exp(grad_coff*label_dis[row, col])
    # label = np.matrix(label_raw).reshape(len(grid.r), 1)
    label = label_raw
    threshhold = threshhold_coff * (label.max()-label.min())+label.min()
    label = np.where(label > threshhold, label - threshhold, 0)
    region = np.where(label > 0, 1, 0)
    return region.flatten(), grad_coff, threshhold_coff


def label_generate(randn_indexr, randn_indexz, value, numgridr, numgridz, grid, grad_coff, threshhold_coff):
    label_raw = np.zeros(len(grid.r)).reshape(numgridr, numgridz)
    label_dis = np.zeros(len(grid.r)).reshape(numgridr, numgridz)
    label_raw[randn_indexr, randn_indexz] = value
    for row in range(numgridr):
        for col in range(numgridz):
            distnace = (abs(row - randn_indexr)**2 + abs(col - randn_indexz)**2)**0.5/3
            label_dis[row, col] = distnace
    for row in range(numgridr):
        for col in range(numgridz):
            label_raw[row, col] = value * np.exp(grad_coff*label_dis[row, col])
    label = np.matrix(label_raw).reshape(len(grid.r), 1)
    threshhold = threshhold_coff * (label.max()-label.min())+label.min()
    label = np.where(label > threshhold, (label - threshhold)/(label.max()-threshhold)*value, 0)

    return label


def input_generate(c_matrix, label):
    input = c_matrix * label
    return input


def load_los_from_path(path):
    los_info = np.loadtxt(path)
    position = [[], []]
    angle = []
    position[0] = los_info[0]
    position[1] = los_info[1]
    angle = los_info[2]
    return position, angle


def load_los(numgridr, numgridz, rdm_point_num, rdm_chord_num,grid_defr,grid_defz):
    position = [[], []]
    angle = []
    r_point = grid_defr[numgridr-1]
    # z_bet = math.floor(numgridz / rdm_point_num)

    if rdm_point_num == 1:
        z_point = grid_defz[numgridz - 1]
        angle_bet = math.floor(abs(180 - 270) / rdm_chord_num)
        for j in range(rdm_chord_num):
            angle.append(180 + j * angle_bet)
            position[0].append(r_point)
            position[1].append(z_point)
    elif rdm_point_num == 2:
        z_point = grid_defz[math.floor((numgridz - 1) / 2)]
        angle_bet = math.floor(abs(135 - 225) / rdm_chord_num)
        for j in range(rdm_chord_num):
            angle.append(135 + j * angle_bet)
            position[0].append(r_point)
            position[1].append(z_point)
    else:
        z_point = grid_defz[0]
        angle_bet = math.floor(abs(90 - 180) / rdm_chord_num)
        for j in range(rdm_chord_num):
            angle.append(90 + j * angle_bet)
            position[0].append(r_point)
            position[1].append(z_point)

    return position, angle

def save_to_dataframe():
    return 0

def plot_data(data):
    max = data.max()
    min = data.min()
    plt.figure()
    plt.contourf(data, cmap='jet')
    plt.colorbar(label='ne')
    ax = plt.gca()
    ax.set_aspect(1.0)
    plt.savefig("."+"/see_label_contour")
    plt.pcolor(data, cmap='jet')
    plt.savefig("."+"/see_label_pcolor")
    plt.show()

def generate_dataset(name,num_region,num_maxvalue_posi,num_value):
    os.makedirs("./phantomdata", exist_ok=True)
    c_matrix_list = []
    region_list = []
    input_list = []
    label_list = []
    for i in range(num_region):
        print(f'start {i} case:')
        size_ratio = random.randint(5, 10)
        size_ratio = size_ratio/(10.0)
        numgridz = random.randint(20, 40)
        numgridr = math.floor(numgridz * size_ratio)
        minr, maxr = 0, numgridr/numgridr
        minz, maxz = 0, numgridz/numgridz
        grid_defr = np.linspace(minr, maxr, numgridr)  # 每列网格的r坐标
        grid_defz = np.linspace(minz, maxz, numgridz)  # 每行网格的z坐标
        center_up = int(0.7 * (numgridz - 1))
        center_down = int(0.3 * (numgridz - 1))
        center_left = int(0.3 * (numgridr - 1))
        center_right = int(0.7 * (numgridr - 1))
        rdm_point_num = random.randint(1, 3)
        rdm_chord_num = random.randint(10, 60)
        los_pos, los_angle = load_los(numgridr, numgridz, rdm_point_num, rdm_chord_num, grid_defr, grid_defz)
        grid = grid_generate(minr, maxr, numgridr, minz, maxz, numgridz)  # 为网格中心的坐标
        c_matrix = get_cmatrix(grid_defr, grid_defz, los_pos, los_angle, grid)
        for k in range(num_maxvalue_posi):
            randn_indexr = random.randint(center_left, center_right)
            randn_indexz = random.randint(center_down, center_up)
            region, grad_coff, threshhold_coff = region_generate(randn_indexr, randn_indexz, numgridr, numgridz, grid)
            print(f'grad_coff = {grad_coff}')
            print(f'threshhold_coff = {threshhold_coff}')
            c_matrix_list.append(c_matrix)
            region_list.append(region)
            for j in range(num_value):
                value = random.uniform(0.5, 1)
                label = label_generate(randn_indexr, randn_indexz, value, numgridr, numgridz, grid, grad_coff,
                                       threshhold_coff)
                # plot_data(label.T[0].reshape(numgridr,numgridz))
                input = input_generate(c_matrix, label)
                new_columns = np.array([[i, numgridr, numgridz]], dtype=np.float64)
                # 将id和网格数量拼接，用于后续可视化
                input_contact = np.column_stack((np.array(input.T), new_columns))
                input_list.append(input_contact[0])
                label_list.append(label.T[0])
    assert len(input_list) == len(label_list), "输入列表和标签列表长度必须相同"

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

    with h5py.File(f"./phantomdata/mini_1_{name[0]}_database.h5", 'a') as data0:
        data_input_group = data0.create_group("x")
        data_label_group = data0.create_group("y")
        data_posi_group = data0.create_group("posi")
        data_region_group = data0.create_group("regi")
        for i in range(len(c_matrix_list)):
            data_posi_group.create_dataset(str(i), data=c_matrix_list[i])
            data_region_group.create_dataset(str(i), data=region_list[i])
        for j in range(len(train_inputs)):
            data_input_group.create_dataset(str(j), data=train_inputs[j])
            data_label_group.create_dataset(str(j), data=train_labels[j])
    with h5py.File(f"./phantomdata/mini_1_{name[1]}_database.h5", 'a') as data1:
        data_input_group = data1.create_group("x")
        data_label_group = data1.create_group("y")
        data_posi_group = data1.create_group("posi")
        data_region_group = data1.create_group("regi")
        for i in range(len(c_matrix_list)):
            data_posi_group.create_dataset(str(i), data=c_matrix_list[i])
            data_region_group.create_dataset(str(i), data=region_list[i])
        for j in range(len(val_inputs)):
            data_input_group.create_dataset(str(j), data=val_inputs[j])
            data_label_group.create_dataset(str(j), data=val_labels[j])
    with h5py.File(f"./phantomdata/mini_1_{name[2]}_database.h5", 'a') as data2:
        data_input_group = data2.create_group("x")
        data_label_group = data2.create_group("y")
        data_posi_group = data2.create_group("posi")
        data_region_group = data2.create_group("regi")
        for i in range(len(c_matrix_list)):
            data_posi_group.create_dataset(str(i), data=c_matrix_list[i])
            data_region_group.create_dataset(str(i), data=region_list[i])
        for j in range(len(test_inputs)):
            data_input_group.create_dataset(str(j), data=test_inputs[j])
            data_label_group.create_dataset(str(j), data=test_labels[j])
    print("finish")
##################################################################


if __name__ == '__main__':
    # 定义列名称
    name = ['train','valid','test']
    num_region = 1
    num_maxvalue_posi = 100
    num_value = 1000
    generate_dataset(name,num_region,num_maxvalue_posi,num_value)




    # with h5py.File("./phantomdata/mini_1_train_database.h5", 'a') as train, \
    #     h5py.File("./phantomdata/mini_1_valid_database.h5", 'a') as valid:
    #     train_input_group = train.create_group("x")
    #     train_label_group = train.create_group("y")
    #     train_posi_group = train.create_group("posi")
    #     train_region_group = train.create_group("regi")
    #     valid_input_group = valid.create_group("x") # 1 * n_d
    #     valid_label_group = valid.create_group("y") # m
    #     valid_posi_group = valid.create_group("posi") # n_d * m
    #     valid_region_group = valid.create_group("regi")# 1 * m
    #     k_train = 0
    #     k_valid = 0
    #     for i in range(20):
    #         print(f'start {i} case:')
    #         input_df = []
    #         label_df = []
    #         size_ratio = random.randint(5, 10)
    #         size_ratio = size_ratio/(10.0)
    #         numgridz = random.randint(20, 40)
    #         numgridr = math.floor(numgridz * size_ratio)
    #         # minr, maxr = 0, numgridr - 1
    #         minr, maxr = 0, numgridr/numgridr
    #         # minz, maxz = 0, numgridz - 1
    #         minz, maxz = 0, numgridz/numgridz
    #         grid_defr = np.linspace(minr, maxr, numgridr)  # 每列网格的r坐标
    #         grid_defz = np.linspace(minz, maxz, numgridz)  # 每行网格的z坐标
    #         center_up = int(0.7 * (numgridz - 1))
    #         center_down = int(0.3 * (numgridz - 1))
    #         center_left = int(0.3 * (numgridr - 1))
    #         center_right = int(0.7 * (numgridr - 1))
    #         randn_indexr = random.randint(center_left, center_right)
    #         randn_indexz = random.randint(center_down, center_up)
    #         rdm_point_num = random.randint(1, 3)
    #         rdm_chord_num = random.randint(10, 60)
    #         los_pos, los_angle = load_los(numgridr, numgridz, rdm_point_num, rdm_chord_num,grid_defr,grid_defz)
    #         grid = grid_generate(minr, maxr, numgridr, minz, maxz, numgridz)  # 为网格中心的坐标
    #         edgepixel_index = find_edgebeams(grid)
    #         c_matrix = get_cmatrix(grid_defr, grid_defz, los_pos, los_angle, grid)
    #         c_matrix_flatten = np.array(c_matrix).flatten()
    #         region, grad_coff, threshhold_coff = region_generate(randn_indexr, randn_indexz, numgridr, numgridz, grid)
    #         # info_flatten = np.hstack((c_matrix_flatten,region.flatten()))
    #         # info_flatten = info_flatten.reshape(1, len(info_flatten))
    #         print(f'grad_coff = {grad_coff}')
    #         print(f'threshhold_coff = {threshhold_coff}')
    #         train_posi_group.create_dataset(str(i), data=c_matrix)
    #         train_region_group.create_dataset(str(i), data=region)
    #         valid_posi_group.create_dataset(str(i), data=c_matrix)
    #         valid_region_group.create_dataset(str(i), data=region)

    #         for j in range(5000):
    #             l = random.randint(0, 9)
    #             value = random.uniform(0.5, 1)
    #             label = label_generate(randn_indexr, randn_indexz, value, numgridr, numgridz, grid, grad_coff,
    #                                    threshhold_coff)
    #             input = input_generate(c_matrix, label)
    #             new_columns = np.array([[i, numgridr, numgridz]], dtype=np.float64)
    #             # 将id和网格数量拼接，用于后续可视化
    #             input_contact = np.column_stack((np.array(input.T), new_columns))

    #             if l > 2:
    #                 train_input_group.create_dataset(str(k_train),data = input_contact[0])
    #                 train_label_group.create_dataset(str(k_train),data = label.T[0])
    #                 k_train +=1
    #             else:
    #                 valid_input_group.create_dataset(str(k_valid), data=input_contact[0])
    #                 valid_label_group.create_dataset(str(k_valid), data=label.T[0])
    #                 k_valid +=1
    #     print("finish")
