from typing import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

INF = 0


def create_data():
    iris = pd.read_csv("lab2\Iris.csv")
    data = iris.values[:, 1:5]
    target = iris.values[:, 5]
    x_train, x_test, y_train, y_test = train_test_split(
        data, target, test_size=0.2, random_state=1024
    )
    return x_train, y_train, x_test, y_test


def draw(x, y, n=0):
    plt.figure(n)
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap="brg")
    plt.scatter(x[:, 2], x[:, 3], c=y, cmap="gist_rainbow")
    plt.show()


def distance(x1, x2, p):
    if p == INF:
        return np.max(np.abs(x1 - x2))
    return np.sum(np.abs((x1 - x2)) ** p) ** (1 / p)


class KNN:
    def __init__(self, k=5, p=2) -> None:
        self.k = k
        self.p = p

    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def predit(self, x):
        dists = [distance(x, np.array(x_x_train), self.p) for x_x_train in self.x_train]
        k_indices = np.argsort(dists)[: self.k]
        k_labs = [self.y_train[i] for i in k_indices]
        most = Counter(k_labs).most_common(1)
        return most

    def score(self, x_test, y_test):
        cnt = 0
        for x, y in zip(x_test, y_test):
            if self.predit(x)[0][0] == y:
                cnt += 1
        return cnt / len(x_test)

    def y_predit(self, x_test):
        y_predit = []
        for x, y in zip(x_test, y_test):
            res = self.predit(x)[0][0]
            y_predit.append(res)
        return np.array(y_predit)


x_train, y_train, x_test, y_test = create_data()

print(len(x_train))

knn = KNN()
knn.fit(x_train, y_train)


knn.p = 1

scores = []
for i in range(1, len(x_train)):
    knn.k = i
    scores.append(knn.score(x_test, y_test))


plt.scatter(list(range(1, len(scores) + 1)), scores, cmap="brg")
plt.xlabel("K")
plt.ylabel("SCORE")
plt.show()
