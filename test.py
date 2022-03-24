from typing import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris


INF = 0


def create_data():
    X = load_iris().data
    y = load_iris().target
    return X, y


def distance(x1, x2, p=1, axis=0):
    if p == INF:
        return np.max(x1 - x2)
    return np.sum(np.abs(x1 - x2) ** p, axis=axis) ** (1 / p)


class KMeans:
    def __init__(self, k=2, t=10) -> None:
        self.k = k
        self.t = t

    def fit(self, X, method=0):
        m = np.shape(X)[0]
        n = np.shape(X)[1]
        if self.k >= m:
            print(f"warning: k({self.k}) >= m({m})")
        self.X = X
        self.labs = np.zeros(m)
        self.centroids = np.zeros((self.k, n))

        if method == 0:
            np.random.shuffle(X)
            self.centroids = X[: self.k]
        elif method == 1:
            self.centroids = X[range(0, self.k)]
        elif method == 2:
            for j in range(n):
                minJ = min(X[:, j])
                rangeJ = float(max(X[:, j]) - minJ)
                self.centroids[:, j] = np.array(minJ + rangeJ * np.random.rand(self.k))

        for t in range(self.t):
            for i, x in enumerate(X):
                dis = distance(x, self.centroids, axis=)1
                self.labs[i] = dis.argmin()
            stable = True
            for i in range(self.k):
                if i not in self.labs:
                    continue
                mean = np.mean(X[self.labs == i], axis=0)
                if any(self.centroids[i] != mean):
                    self.centroids[i] = mean
                    stable = False
            if stable:
                break

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
                np.sum((self.X[self.labs == i] - self.centroids[i]) ** 2, axis=1)
            )
        return wss


group, labs = create_data()
km = KMeans()
wss = []
for k in range(2, 15):
    km.k = k
    km.t = 20
    km.fit(group, method=0)
    w = km.wss()
    print(km.wss())
    wss.append(w)
print(wss)

plt.figure(0)
plt.plot(range(2, len(wss) + 2), wss)
plt.xlabel("K")
plt.ylabel("WSS")
plt.show()
