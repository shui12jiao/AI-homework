from pydoc import cram
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris


def iris_type(s):
    it = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
    return it[s]


def create_data():
    data = load_iris().data[:, :2]
    target = load_iris().target
    x_train, x_test, y_train, y_test = train_test_split(
        data, target, random_state=1, train_size=0.6
    )
    return x_train, y_train, x_test, y_test, data, target


def show_accuracy(y_hat, y_test, param):
    pass


def laplace(X, Y):
    pass
    # np.exp(-np.abs)
    # return np.dot(X, Y.T)
    # # for j in range(m):
    # #     deltaRow = X[j, :] - A
    # #     K[j] = deltaRow * deltaRow.T
    # #     K[j] = sqrt(K[j])
    # # K = exp(-K / kTup[1])


def draw(x, s):
    # 可以通过修改 kernel 参数来实现不同核函数的验证
    print("decision_function:\n", s.clf.decision_function(s.x_train))
    print("\npredict:\n", s.clf.predict(s.x_train))
    x1_min, x1_max = x[:, 0].min(), x[:, 0].max()  # 第 0 列的范围
    x2_min, x2_max = x[:, 1].min(), x[:, 1].max()  # 第 1 列的范围
    x1, x2 = np.mgrid[x1_min:x1_max:200j, x2_min:x2_max:200j]  # 生成网格采样点
    grid_test = np.stack((x1.flat, x2.flat), axis=1)  # 测试点
    mpl.rcParams["font.sans-serif"] = ["SimHei"]
    mpl.rcParams["axes.unicode_minus"] = False
    cm_light = mpl.colors.ListedColormap(["#A0FFA0", "#FFA0A0", "#A0A0FF"])
    cm_dark = mpl.colors.ListedColormap(["g", "r", "b"])
    # print 'grid_test = \n', grid_test
    grid_hat = s.clf.predict(grid_test)  # 预测分类值
    grid_hat = grid_hat.reshape(x1.shape)  # 使之与输入的形状相同
    alpha = 0.5

    plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light)  # 预测值的显示
    # plt.scatter(x[:, 0], x[:, 1], c=y, edgecolors='k', s=50, cmap=cm_dark) # 样本
    plt.plot(x[:, 0], x[:, 1], "o", alpha=alpha, color="blue", markeredgecolor="k")
    plt.scatter(
        x_test[:, 0], x_test[:, 1], s=120, facecolors="none", zorder=10
    )  # 圈中测试集样本
    plt.xlabel("花萼长度", fontsize=13)
    plt.ylabel("花萼宽度", fontsize=13)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.title("鸢尾花 SVM 二特征分类", fontsize=15)
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

    def fit_score(self):
        self.y_hat = self.clf.predict(x_train)
        return self.clf.score(x_train, y_train), self.y_hat

    def predict(self, x_test):
        return self.clf.predict(x_test)

    def score(self, x_test, y_test):
        return self.clf.score(x_test, y_test)


x_train, y_train, x_test, y_test, x, y = create_data()
kernels = ("linear", "poly", "rbf", "sigmoid", laplace)  # 线性、多项式、高斯、Sigmoid、

svmCase = SVM()
svmCase.fit(x_train, y_train, kernel=kernels[2])
print("训练集：", svmCase.fit_score())
print(
    "测试集",
    svmCase.score(x_test, y_test),
    svmCase.predict(x_test),
)
# draw(x, svmCase)
