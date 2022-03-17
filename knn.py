from typing import Counter
import numpy as np


def create_data():
    group = np.array([[1.5, 2.6, 4], [3.9, 3.1, 6.6], [5.1, 1.2, 0.7],
                      [8.0, 8.9, 6.8], [1.0, 2.2, 3.3], [2.2, 2.1, 2.6], [3.4, 3.7, 7.7], [9.1, 8.1, 0.5]])
    labels = ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B']
    return group, labels


def distance(x1, x2, p):
    return np.sum((x1 - x2)**p)**(1/p)


class KNN:

    def __init__(self, k=5, p=2) -> None:
        self.k = k
        self.p = p

    def fit(self, group, labels):
        self.group = group
        self.labels = labels

    def predit(self, x):
        dists = [distance(x, np.array(x_group), self.p)
                 for x_group in self.group]
        k_indices = np.argsort(dists)[:self.k]
        k_labs = [self.labels[i] for i in k_indices]
        most = Counter(k_labs).most_common(1)
        return most


knn = KNN()
group, labels = create_data()
knn.fit(group, labels)
print(knn.predit(np.array([2.2, 2.4, 5.8])))
knn.k = 3
print(knn.predit(np.array([2.2, 2.4, 5.8])))
