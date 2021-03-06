from typing import Counter
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

INF = 0


def create_data():
    data = np.array(
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
    target = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    x_train, x_test, y_train, y_test = train_test_split(
        data, target, test_size=0.2, random_state=1024
    )
    return x_train, y_train, x_test, y_test


def draw(x, y, n=0):
    plt.figure(n)
    # plt.scatter(x[:, 0], x[:, 1], c=y, cmap="brg")
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap="rainbow")
    plt.show()


def distance(x1, x2, p):
    if p == INF:
        return np.max(x1 - x2)
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


knn = KNN()
x_train, y_train, x_test, y_test = create_data()
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
