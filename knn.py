from cProfile import label
import numpy as np


def create_data():
    group = np.array([[1.5, 2.6, 4, 4], [3.9, 3.1, 6.6], [5.1, 1.2, 0.7],
                      [8.0, 8.9, 6.8]])
    label = ['A', 'A', 'B', 'B']
    return group, label


def distance(d1, d2, n, p):
    res = 0
    for i in range(n):
        res += abs(d1[i] - d2[i])**p
    res = res**(1 / p)


class KNN:

    def __init__(self, K, array, label) -> None:
        self.K = K
        self.array = array
        self.label = label
