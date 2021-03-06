from pydoc import cram
from re import X
from sklearn import svm
from numpy import *
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import os


def trainingDigits():
    fileList = os.listdir("lab4/trainingDigits")
    length = len(fileList)
    x, y = [], []

    def read_img(filename):
        img = zeros(1024)
        fr = open(filename)
        for i in range(32):
            line = fr.readline()
        for j in range(32):
            img[32 * i + j] = int(line[j])
        return img

    for f in fileList[0:20] + fileList[230:250]:
        y.append(int(f[0]))
        x.append(read_img("lab4/trainingDigits/" + f))
    # return np.array(x).reshape((len(x), 1024)), np.array(y).reshape((len(x), 1024))
    x, y = array(x), array(y)
    for i, v in enumerate(y):
        if v == 0:
            y[i] = -1
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, random_state=1, train_size=0.6
    )
    return (
        mat(x_train),
        mat(y_train).transpose(),
        mat(x_test),
        mat(y_test).transpose(),
        mat(x),
        mat(y),
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
        mat(x_train),
        mat(y_train).transpose(),
        mat(x_test),
        mat(y_test).transpose(),
        mat(dataMat),
        mat(labelMat),
    )


def create_data():
    data = load_iris().data[:, :2]
    target = load_iris().target

    index = array([], dtype=int)
    for i in range(shape(target)[0]):
        if target[i] == 2:
            index = append(index, i)
        elif target[i] == 0:
            target[i] = -1
    data = delete(data, index, 0)
    target = delete(target, index, 0)
    x_train, x_test, y_train, y_test = train_test_split(
        data, target, random_state=1, train_size=0.6
    )
    return (
        mat(x_train),
        mat(y_train).transpose(),
        mat(x_test),
        mat(y_test).transpose(),
        mat(data),
        mat(target),
    )


# ???????????????????????????????????????
def kernelTrans(X, A, kTup):
    m, n = shape(X)
    K = mat(zeros((m, 1)))
    if kTup[0] == "linear":  # ???????????????
        K = X * A.T
    elif kTup[0] == "rbf":  # ?????????
        for j in range(m):
            deltaRow = X[j, :] - A
        K[j] = deltaRow * deltaRow.T
        K = exp(K / (-1 * kTup[1] ** 2))
    elif kTup[0] == "laplace":  # ???????????????
        for j in range(m):
            deltaRow = X[j, :] - A
            K[j] = deltaRow * deltaRow.T
            K[j] = sqrt(K[j])
        K = exp(-K / kTup[1])
    elif kTup[0] == "poly":  # ????????????
        K = X * A.T
        for j in range(m):
            K[j] = K[j] ** kTup[1]
    elif kTup[0] == "sigmoid":  # Sigmoid ???
        K = X * A.T
        for j in range(m):
            K[j] = tanh(kTup[1] * K[j] + kTup[2])
    else:
        raise NameError("???????????????????????? -- ?????????????????????")
    return K


def show_accuracy(y_hat, y_test, param):
    pass


def draw(x, s, xname, yname, topic):
    # ?????????????????? kernel ???????????????????????????????????????
    # print("decision_function:\n", s.decision_function(s.x_train))
    # print("\npredict:\n", s.predict(s.x_train))
    x1_min, x1_max = x[:, 0].min(), x[:, 0].max()  # ??? 0 ????????????
    x2_min, x2_max = x[:, 1].min(), x[:, 1].max()  # ??? 1 ????????????
    x1, x2 = mgrid[x1_min:x1_max:200j, x2_min:x2_max:200j]  # ?????????????????????
    grid_test = stack((x1.flat, x2.flat), axis=1)  # ?????????
    mpl.rcParams["font.sans-serif"] = ["SimHei"]
    mpl.rcParams["axes.unicode_minus"] = False
    cm_light = mpl.colors.ListedColormap(["#A0FFA0", "#FFA0A0", "#A0A0FF"])
    cm_dark = mpl.colors.ListedColormap(["g", "r", "b"])
    # print 'grid_test = \n', grid_test
    grid_hat = s.predict(grid_test)  # ???????????????
    grid_hat = grid_hat.reshape(x1.shape)  # ??????????????????????????????
    alpha = 0.5
    plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light)  # ??????????????????
    plt.scatter(x[:, 0], x[:, 1], c=y, edgecolors="k", s=50, cmap=cm_dark)  # ??????
    plt.scatter(
        # array(x_test)[:, 0], array(x_test)[:, 1], s=120, facecolors="none", zorder=10
        array(x_train)[:, 0],
        array(x_train)[:, 1],
        s=120,
        facecolors="none",
        zorder=10,
    )  # ?????????????????????
    plt.xlabel(xname, fontsize=13)
    plt.ylabel(yname, fontsize=13)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.title(topic, fontsize=15)
    plt.show()


# ????????????????????????????????????????????????
def plot_point(filename, alphas, dataMat):
    filename = filename
    fr = open(filename)
    X1 = []
    y1 = []
    X2 = []
    y2 = []
    for line in fr.readlines():
        lineArr = line.strip().split("\t")
        if float(lineArr[-1]) == 1:
            X1.append(lineArr[0])
            y1.append(lineArr[1])
        elif float(lineArr[-1]) == -1:
            X2.append(lineArr[0])
            y2.append(lineArr[1])
    plt.scatter(X1[:], y1[:], c="y", s=50)
    plt.scatter(X2[:], y2[:], c="b", s=50)
    for i, alpha in enumerate(alphas):
        if abs(alpha) > 0:
            xy = dataMat.getA()[i]
            plt.scatter(
                xy[0],
                xy[1],
                s=100,
                # c="",
                alpha=0.5,
                linewidth=1.5,
                edgecolor="red",
            )
    plt.show()


class entity:
    # Initialize the structure with the parameters
    def __init__(self, dataMatIn, classLabels, C, toler, kTup=("linear", 0)):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.n = shape(dataMatIn)[1]
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0
        self.eCache = mat(zeros((self.m, 2)))  # first column is valid flag
        self.K = mat(zeros((self.m, self.m)))
        for i in range(self.m):
            self.K[:, i] = kernelTrans(self.X, self.X[i, :], kTup)


class SMO:
    def __init__(self, entity, maxIter=10000):
        # self.ent = self.entity(mat(dataMatIn), mat(classLabels).transpose(), C, toler, kTup)
        self.maxIter = maxIter
        self.ent = entity

    def smop(self):
        iter = 0
        entireSet = True
        alphaPairsChanged = 0
        # ???????????? 1???iter ??????????????????????????? && ???????????? 2????????????????????????????????????????????????????????????
        # ?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
        while iter < self.maxIter and ((alphaPairsChanged > 0) or entireSet):
            alphaPairsChanged = 0
            # ??????????????????
            if entireSet:
                for i in range(self.ent.m):
                    alphaPairsChanged += self.__innerL(i)
                # print(
                #     "fullSet, iter:",
                #     iter,
                #     ", i:",
                #     i,
                #     ", pairs changed:",
                #     alphaPairsChanged,
                # )
                iter += 1

            # ??????????????????
            else:
                # ??????????????????????????????
                nonBoundIs = nonzero(
                    (self.ent.alphas.A > 0) * (self.ent.alphas.A < self.ent.C)
                )[0]
                for i in nonBoundIs:
                    alphaPairsChanged += self.__innerL(i)
                # print(
                #     "non-bound, iter:",
                #     iter,
                #     ", i:",
                #     i,
                #     ", pairs changed: ",
                #     alphaPairsChanged,
                # )
                iter += 1

            # ?????????????????????????????????????????????????????????????????????????????????
            if entireSet:
                entireSet = False
            # ?????????????????????????????????????????????????????????????????????????????????
            elif alphaPairsChanged == 0:
                entireSet = True
            # print("iteration number: ", iter)
        return self.ent.b, self.ent.alphas

    # selectJrand ???????????? i???????????? i ?????????????????????????????? alpha ?????????.
    def __selectJrand(self, i, m):
        j = i  # ?????? alpha ???????????????
        while j == i:
            j = int(random.uniform(0, m))
        return j

    # ??????????????? yi????????? alphaj ?????????
    def __clipAlpha(self, aj, H, L):
        if aj > H:
            aj = H
        if aj < L:
            aj = L
        return aj

    # #?????????????????? E = f(xi) - yi ????????????????????? E
    def __calcEk(self, k):
        fXk = mat(
            multiply(self.ent.alphas, self.ent.labelMat).T * self.ent.K[:, k]
            + self.ent.b
        )
        Ek = fXk - float(self.ent.labelMat[k])
        return Ek

    # # ??????????????? alpha ?????????????????? alpha ???
    def __selectJ(self, i, Ei):
        maxK = -1
        maxDeltaE = 0
        Ej = 0
        self.ent.eCache[i] = [1, Ei]
        validEcacheList = nonzero(self.ent.eCache[:, 0].A)[0]
        # print(validEcacheList)
        if (len(validEcacheList)) > 1:
            for k in validEcacheList:
                if k == i:
                    continue
                Ek = self.__calcEk(k)
                deltaE = abs(Ei - Ek)
                if deltaE > maxDeltaE:
                    maxK = k
                    maxDeltaE = deltaE
                    Ej = Ek
            return maxK, Ej
        else:  # in this case (first time around) we don't have any valid eCache values
            j = self.__selectJrand(i, self.ent.m)
            Ej = self.__calcEk(j)
        return j, Ej

    # ????????????
    def __updateEk(self, k):
        Ek = self.__calcEk(k)
        self.ent.eCache[k] = [1, Ek]

    # SMO ????????????????????????????????????????????????
    # ????????? self.ent.X[i, :] * self.ent.X[j, :].T ????????? self.ent.K[j, j]???????????????????????????????????????????????????
    def __innerL(self, i):
        Ei = self.__calcEk(i)
        if (
            (self.ent.labelMat[i] * Ei < -self.ent.tol)
            and (self.ent.alphas[i] < self.ent.C)
        ) or ((self.ent.labelMat[i] * Ei > self.ent.tol) and (self.ent.alphas[i] > 0)):
            j, Ej = self.__selectJ(i, Ei)
            alphaIold = self.ent.alphas[i].copy()
            alphaJold = self.ent.alphas[j].copy()
            if self.ent.labelMat[i] != self.ent.labelMat[j]:
                L = max(0, self.ent.alphas[j] - self.ent.alphas[i])
                H = min(
                    self.ent.C, self.ent.C + self.ent.alphas[j] - self.ent.alphas[i]
                )
            else:
                L = max(0, self.ent.alphas[i] + self.ent.alphas[j] - self.ent.C)
                H = min(self.ent.C, self.ent.alphas[i] + self.ent.alphas[j])
            if L == H:
                # print("L == H")
                return 0
            # eta ??? alpha[j]??????????????????
            eta = 2.0 * self.ent.K[i, j] - self.ent.K[i, i] - self.ent.K[j, j]
            # eta >= 0 ??????????????????????????????????????????????????????????????????????????????????????????????????????
            if eta >= 0:
                # print("eta >= 0")
                return 0
            self.ent.alphas[j] -= self.ent.labelMat[j] * (Ei - Ej) / eta
            self.ent.alphas[j] = self.__clipAlpha(self.ent.alphas[j], H, L)
            self.__updateEk(j)
            # ????????????????????????????????????????????????????????????????????????
            if abs(self.ent.alphas[j] - alphaJold) < 1e-5:
                # print("j not moving enough")
                return 0
            self.ent.alphas[i] += (
                self.ent.labelMat[j]
                * self.ent.labelMat[i]
                * (alphaJold - self.ent.alphas[j])
            )
            self.__updateEk(i)
            b1 = (
                self.ent.b
                - Ei
                - self.ent.labelMat[i]
                * (self.ent.alphas[i] - alphaIold)
                * self.ent.K[i, i]
                - self.ent.labelMat[j]
                * (self.ent.alphas[j] - alphaJold)
                * self.ent.K[i, j]
            )
            b2 = (
                self.ent.b
                - Ej
                - self.ent.labelMat[i]
                * (self.ent.alphas[i] - alphaIold)
                * self.ent.K[i, j]
                - self.ent.labelMat[j]
                * (self.ent.alphas[j] - alphaJold)
                * self.ent.K[j, j]
            )
            if (0 < self.ent.alphas[i]) and (self.ent.C > self.ent.alphas[i]):
                self.ent.b = b1
            elif (0 < self.ent.alphas[j]) and (self.ent.C > self.ent.alphas[j]):
                self.ent.b = b2
            else:
                self.ent.b = (b1 + b2) / 2.0
            return 1
        else:
            return 0


class SVM:
    def __init__(self, entity):
        self.ent = entity

    def fit(self):
        smo = SMO(self.ent)
        smo.smop()
        self.w = self.__calculateW()

    def predict(self, x):
        x = x
        wt = self.w
        m = shape(x)[0]
        res = empty(m)
        for i in range(m):
            tmp = float(x[i, :] * wt) + self.ent.b
            if tmp >= 0:
                res[i] = 1
            else:
                res[i] = -1
        return res

    def score(self, x, y):
        y = array(y)
        res = self.predict(x)
        score = 0
        for i in range(shape(x)[0]):
            if y[i] == res[i]:
                score += 1
        return score / shape(res)[0]

    # ??????????????? alpha ????????? W ??????
    def __calculateW(self):
        # alphas, dataMat, labelMat = array(alphas), array(dataArr), array(labelArr)
        alphas, dataMat, labelMat = (
            self.ent.alphas,
            self.ent.X,
            self.ent.labelMat,
        )
        sum = 0
        for i in range(shape(dataMat)[0]):
            sum += multiply(alphas[i] * labelMat[i], dataMat[i, :].T)
        # print(sum)
        return sum


# ????????????
if __name__ == "__main__":
    # testRbf()
    kernels = ("linear", "poly", "rbf", "laplace", "sigmoid")  # ??????????????????????????????Sigmoid???
    # x_train, y_train, x_test, y_test, x, y = create_data()
    # x_train, y_train, x_test, y_test, x, y = create_data_("lab4/testSetRBF2.txt")

    x_train, y_train, x_test, y_test, x, y = trainingDigits()

    # print("W:", svmCase.w, "\nAlphas:\n", svmCase.ent.alphas)

    for i in range(4):
        ent = entity(x_train, y_train, 2, 0.0001, (kernels[i], 5))
        svmCase = SVM(ent)
        svmCase.fit()
        print(kernels[i])
        print(
            "??????????????????",
            svmCase.score(x_train, y_train),
            # svmCase.predict(x_train),
        )

        print(
            "??????????????????",
            svmCase.score(x_test, y_test),
            # svmCase.predict(x_test),
        )

        # print(
        #     "??????????????????",
        #     svmCase.score(x, y),
        #     # svmCase.predict(x_test),
        # )
