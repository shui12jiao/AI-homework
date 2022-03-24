from typing import Counter
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


INF = 0


def create_data():
    X = np.array(
        [
            [0.697, 0.460],
            [0.774, 0.376],
            [0.634, 0.264],
            [0.608, 0.318],
            [0.556, 0.215],
            [0.403, 0.237],
            [0.481, 0.149],
            [0.437, 0.211],
            [0.666, 0.091],
            [0.243, 0.267],
            [0.245, 0.057],
            [0.343, 0.099],
            [0.639, 0.161],
            [0.657, 0.198],
            [0.360, 0.370],
            [0.593, 0.042],
            [0.719, 0.103],
        ]
    )
    y = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    return X, y


def distance(x1, x2, p=2, axis=0):
    if p == INF:
        return np.max(x1 - x2)
    return np.sum(np.abs(x1 - x2) ** p, axis=axis) ** (1 / p)


class KMeans:
    def __init__(self, k=2, t=10) -> None:
        self.k = k
        self.t = t

    def fit(self, X):
        self.X = X
        n = np.shape(X)[1]
        self.labs = np.zeros(np.shape(X)[0])
        self.centroids = np.zeros((self.k, n))
        for j in range(n):
            minJ = min(X[:, j])
            rangeJ = float(max(X[:, j]) - minJ)
            self.centroids[:, j] = np.array(minJ + rangeJ * np.random.rand(self.k))

        for t in range(self.t):
            for i, x in enumerate(X):
                dis = distance(x, self.centroids, axis=1)
                self.labs[i] = dis.argmin()
            for i in range(self.k):
                self.centroids = np.mean(X[self.labs == i], axis=0)
        return self.centroids, self.labs

    def predit(self, X):
        labs = np.zeros(np.shape(X)[0])
        for i, x in enumerate(X):
            dis = distance(x, self.centroids, axis=1)
            labs[i] = dis.argmin()
        return labs

    def wss(self):
        wss = 0
        for i in range(self.k):
            wss += np.sum(
                np.sum(np.abs(self.X[self.labs == 1] - self.centroids[i]) ** 2, axis=1)
            )
        return wss


group, labs = create_data()

km = KMeans()
km.fit(group)
print(km.wss())
