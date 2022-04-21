import enum
from turtle import shape
import numpy as np
import struct
import math

# 初始化输入输出层链接权值
def initCompetition(n, m, w, h):
    array = np.random.random(size=n * m * w * h)
    com_weight = array.reshape(n, m, w, h)
    return com_weight


# 计算欧氏距离用于归一化处理
def cal2NF(X):
    return sum(np.multiply(X, X)) ** 0.5


# 将数据归一化
def normalize_data(train_data):
    for x in train_data:
        for data in x:
            two_NF = cal2NF(data)
            for i in range(len(data)):
                data[i] = data[i] / two_NF
    # 返回归一化处理之后的数据，数据结构不变
    return train_data


# 将权值归一化
def nomalize_weight(com_weight):
    for x in com_weight:
        for data in x:
            two_NF = cal2NF(data)
            for i in range(len(data)):
                data[i] = data[i] / two_NF
    return com_weight


# 得到获胜神经元索引
def getWinner(data, com_weight):
    mark_n, mark_m = 0, 0
    min = float("inf")
    n, m = np.shape(com_weight)[0], np.shape(com_weight)[1]

    for i in range(n):
        for j in range(m):
            res = np.sum(np.dot(data, com_weight[i][j]))
            if res < min:
                min = res
                mark_n, mark_m = i, j
    # 返回该神经元的行值和列值 mark_n 为行值，mark_m 为列值
    return mark_n, mark_m


# 得到获胜神经元周围的兴奋神经元的索引
def getNeighbor(n, m, N_neighbor, com_weight):
    res = []
    nn, mm, ww, hh = np.shape(com_weight)
    for i in range(nn):
        for j in range(mm):
            N = int(((i - n) ** 2 + (j - m) ** 2) ** 0.5)
            if N <= N_neighbor:
                res.append((i, j, N))
    return res


# 学习率 与迭代次数和拓扑距离相关
def eta(t, N):
    return (0.3 / (t + 1)) * (math.e ** (-N))


# som
def som(train_data, train_label, com_weight, T, N_neighbor):
    for t in range(T - 1):
        print("epoch:" + str(t))
        com_weight = nomalize_weight(com_weight)
        for data in train_data:
            n, m = getWinner(data, com_weight)
            neighbor = getNeighbor(n, m, N_neighbor, com_weight)
            for x in neighbor:
                j_n = x[0]
                j_m = x[1]
                N = x[2]
                com_weight[j_n][j_m] = com_weight[j_n][j_m] + eta(t, N) * (
                    data - com_weight[j_n][j_m]
                )
            N_neighbor = N_neighbor - (t + 1) / 200
    return com_weight


# 为每个神经元打上标签
def create_labels(com_weight):
    N, M, w, h = np.shape(com_weight)
    belong = {}
    for i in range(len(train_data)):
        n, m = getWinner(train_data[i], com_weight)
        key = n * M + m
        if key in belong:
            belong[key].append(train_labels[i])
        else:
            belong[key] = []
            belong[key].append(train_labels[i])
    labels = {}

    for i in belong.keys():
        tags = belong.get(i)
        numOfTag = np.zeros(10)
        for num in tags:
            num = int(num)
            numOfTag[num] += 1
        labels[i] = np.argmax(numOfTag)
    return labels, com_weight


def test(labels, weight, test_data):
    # weight = som(train_data, train_labels, com_weight, T, N_neighbor)
    # labels, weight = create_labels(weight)
    # test(labels, weight, test_image)
    _, M, _, _ = np.shape(com_weight)
    for i, data in enumerate(test_data):
        n, m = getWinner(data, weight)
        i = n * M + m
        print(labels.get(i))
    # 数据的标签与最近的神经元相同


# 训练集文件
train_images_idx3_ubyte_file = "lab6/MNIST/train-images.idx3-ubyte"
# 训练集标签文件
train_labels_idx1_ubyte_file = "lab6/MNIST/train-labels.idx1-ubyte"
# 测试集文件
test_images_idx3_ubyte_file = "lab6/MNIST/t10k-images.idx3-ubyte"
# 测试集标签文件
test_labels_idx1_ubyte_file = "lab6/MNIST/t10k-labels.idx1-ubyte"


def decode_idx3_ubyte(idx3_ubyte_file):
    """
    解析 idx3 文件的通用函数
    :param idx3_ubyte_file: idx3 文件路径
    :return: 数据集
    """
    # 读取二进制数据
    bin_data = open(idx3_ubyte_file, "rb").read()
    # 解析文件头信息，依次为魔数、图片数量、每张图片高、每张图片宽
    offset = 0
    fmt_header = ">iiii"
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(
        fmt_header, bin_data, offset
    )
    # 解析数据集
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)
    print(offset)
    fmt_image = ">" + str(image_size) + "B"

    print(fmt_image, offset, struct.calcsize(fmt_image))
    images = np.empty((num_images, num_rows, num_cols))
    for i in range(num_images):
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape(
            (num_rows, num_cols)
        )
        offset += struct.calcsize(fmt_image)
    return images


def decode_idx1_ubyte(idx1_ubyte_file):
    """
    解析 idx1 文件的通用函数
    :param idx1_ubyte_file: idx1 文件路径
    :return: 数据集
    """
    # 读取二进制数据
    bin_data = open(idx1_ubyte_file, "rb").read()
    # 解析文件头信息
    offset = 0
    fmt_header = ">ii"
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    # 解析数据集
    offset += struct.calcsize(fmt_header)
    fmt_image = ">B"
    labels = np.empty(num_images)
    for i in range(num_images):
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels


# 自行设置参数：SOM 网络 size：（M，N） 迭代次数：T 近邻范围：N_neighbor
if __name__ == "__main__":
    # 为降低运算量，这里只加载 500 条数据
    train_datas = decode_idx3_ubyte(train_images_idx3_ubyte_file)[0:500]
    train_labels = decode_idx1_ubyte(train_labels_idx1_ubyte_file)[0:500]
    test_image = decode_idx3_ubyte(test_images_idx3_ubyte_file)[10]
    test_label = decode_idx1_ubyte(test_labels_idx1_ubyte_file)[10]
    T = 50
    N_neighbor = 6
    train_data = normalize_data(train_datas)
    com_weight = initCompetition(8, 8, 28, 28)
    # labels : 每个神经元的标签
    weight = som(train_data, train_labels, com_weight, T, N_neighbor)
    labels, weight = create_labels(weight)
    test(labels, weight, test_image)
    print(test_label)
