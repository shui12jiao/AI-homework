from pydoc import cram
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import os, math


def trainingDigits():
    fileList = os.listdir("lab4/trainingDigits")
    length = len(fileList)
    x, y = [], []

    def read_img(filename):
        img = np.zeros(1024)
        fr = open(filename)
        for i in range(32):
            line = fr.readline()
        for j in range(32):
            img[32 * i + j] = int(line[j])
        return img

    for f in fileList[0:393]:
        y.append(int(f[0]))
        x.append(read_img("lab4/trainingDigits/" + f))
    # return np.array(x).reshape((len(x), 1024)), np.array(y).reshape((len(x), 1024))
    x, y = np.array(x), np.array(y)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, random_state=1, train_size=0.6
    )
    return (
        x_train,
        y_train,
        x_test,
        y_test,
        x,
        y,
    )


def create_data_(fileName):
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split("\t")
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))

    x_train, x_test, y_train, y_test = train_test_split(
        dataMat, labelMat, random_state=1, train_size=0.6
    )
    return (
        np.array(x_train),
        np.array(y_train).transpose(),
        np.array(x_test),
        np.array(y_test).transpose(),
        np.array(dataMat),
        np.array(labelMat),
    )


def create_data():
    data = load_iris().data[:, :2]
    target = load_iris().target

    index = np.array([], dtype=int)
    for i in range(np.shape(target)[0]):
        if target[i] == 2:
            index = np.append(index, i)
        elif target[i] == 0:
            target[i] = -1
    data = np.delete(data, index, 0)
    target = np.delete(target, index, 0)
    x_train, x_test, y_train, y_test = train_test_split(
        data, target, random_state=1, train_size=0.6
    )
    return (
        x_train,
        y_train,
        x_test,
        y_test,
        data,
        target,
    )


def show_accuracy(y_hat, y_test, param):
    pass


def laplace(X, Y):
    K = np.zeros((len(X), len(Y)), dtype=np.float)
    for i in range(len(X)):
        for j in range(len(Y)):
            K[i][j] = math.exp(-math.sqrt(np.dot(X[i] - Y[j], (X[i] - Y[j]).T)) / 2)
    return K


def draw(x, s, xname, yname, topic):
    # ?????????????????? kernel ???????????????????????????????????????
    # print("decision_function:\n", s.clf.decision_function(s.x_train))
    # print("\npredict:\n", s.clf.predict(s.x_train))
    x1_min, x1_max = x[:, 0].min(), x[:, 0].max()  # ??? 0 ????????????
    x2_min, x2_max = x[:, 1].min(), x[:, 1].max()  # ??? 1 ????????????
    x1, x2 = np.mgrid[x1_min:x1_max:200j, x2_min:x2_max:200j]  # ?????????????????????
    grid_test = np.stack((x1.flat, x2.flat), axis=1)  # ?????????
    mpl.rcParams["font.sans-serif"] = ["SimHei"]
    mpl.rcParams["axes.unicode_minus"] = False
    cm_light = mpl.colors.ListedColormap(["#A0FFA0", "#FFA0A0", "#A0A0FF"])
    cm_dark = mpl.colors.ListedColormap(["g", "r", "b"])
    # print 'grid_test = \n', grid_test
    grid_hat = s.clf.predict(grid_test)  # ???????????????
    grid_hat = grid_hat.reshape(x1.shape)  # ??????????????????????????????
    alpha = 0.5

    plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light)  # ??????????????????
    # plt.scatter(x[:, 0], x[:, 1], c=y, edgecolors='k', s=50, cmap=cm_dark) # ??????
    plt.plot(x[:, 0], x[:, 1], "o", alpha=alpha, color="blue", markeredgecolor="k")
    plt.scatter(
        x_train[:, 0], x_train[:, 1], s=120, facecolors="none", zorder=10
    )  # ?????????????????????
    plt.xlabel(xname, fontsize=13)
    plt.ylabel(yname, fontsize=13)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.title(topic, fontsize=15)
    plt.show()


class SVM:
    def __init__(self) -> None:
        pass

    def fit(self, x_train, y_train, kernel="linear", C=1, gamma="scale"):
        self.x_train = x_train
        self.y_train = y_train
        self.clf = svm.SVC(
            C=C, kernel=kernel, gamma=gamma, decision_function_shape="ovr"
        )
        self.clf.fit(x_train, y_train)

    def predict(self, x_test):
        return self.clf.predict(x_test)

    def score(self, x_test, y_test):
        return self.clf.score(x_test, y_test)


x_train, y_train, x_test, y_test, x, y = trainingDigits()
# x_train, y_train, x_test, y_test, x, y = create_data()
# x_train, y_train, x_test, y_test, x, y = create_data_("lab4/testSetRBF.txt")
kernels = ("linear", "poly", "rbf", "sigmoid", laplace)  # ??????????????????????????????Sigmoid???

svmCase = SVM()

for i in range(0, 4):
    print(kernels[i])
    svmCase.fit(x_train, y_train, kernel=kernels[i])

    print(
        "??????????????????",
        svmCase.score(x_train, y_train),
        # svmCase.predict(x_train),
    )
    print(
        "??????????????????",
        svmCase.score(x_test, y_test),
        # svmCase.predict(x_train),
    )
    # draw(x, svmCase, "0", "1", "trainingDigits")
    # draw(x, svmCase, "X", "Y", "testSetRBF.txt")
