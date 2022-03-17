from typing import Counter
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def create_data():
    iris = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=77)
    return x_train, y_train, x_test, y_test


def draw(x, y, n=0):
    plt.figure(n)
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap='brg')
    plt.scatter(x[:, 2], x[:, 3], c=y, cmap='gist_rainbow')
    plt.show()


def distance(x1, x2, p):
    return np.sum((x1 - x2)**p)**(1 / p)


class KNN:

    def __init__(self, k=5, p=2) -> None:
        self.k = k
        self.p = p

    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def predit(self, x):
        dists = [
            distance(x, np.array(x_x_train), self.p)
            for x_x_train in self.x_train
        ]
        k_indices = np.argsort(dists)[:self.k]
        k_labs = [self.y_train[i] for i in k_indices]
        most = Counter(k_labs).most_common(1)
        return most

    def score(self, x_test, y_test):
        cnt = 0
        for x, y in zip(x_test, y_test):
            if self.predit(x)[0][0] == y:
                cnt += 1
        return cnt/len(x_test)

    def y_predit(self, x_test):
        y_predit = []
        for x, y in zip(x_test, y_test):
            res = self.predit(x)[0][0]
            y_predit.append(res)
        return np.array(y_predit)


x_train, y_train, x_test, y_test = create_data()
# draw(x_train, y_train, 1)
# draw(x_test, y_test, 1)
# draw(x_test, y_test, 1)

knn = KNN(k=1)
knn.fit(x_train, y_train)

# print(knn.score(x_test, y_test))
# knn.k = 5
# print(knn.score(x_test, y_test))
# knn.p = 1
# print(knn.score(x_test, y_test))

print(y_test)
y_predit = knn.y_predit(x_test)
print(y_predit)
