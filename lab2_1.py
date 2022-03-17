from typing import Counter
import numpy as np


def create_data():
    x_train = np.array([[0.697, 0.460], [0.774, 0.376], [0.634, 0.264],
                        [0.608, 0.318], [0.556, 0.215], [0.403, 0.237],
                        [0.481, 0.149], [0.437, 0.211], [0.666, 0.091],
                        [0.243, 0.267], [0.245, 0.057], [0.343, 0.099],
                        [0.639, 0.161], [0.657, 0.198], [0.360, 0.370],
                        [0.593, 0.042], [0.719, 0.103]])
    y_train = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    return x_train, y_train


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


knn = KNN()
x_train, y_train = create_data()
knn.fit(x_train, y_train)
print(knn.predit(np.array([2.2, 2.4, 5.8])))
knn.k = 3
print(knn.predit(np.array([2.2, 2.4, 5.8])))
