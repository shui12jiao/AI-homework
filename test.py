from cmath import inf
from typing import Counter
from matplotlib.font_manager import FontProperties
from numpy import *
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris


# # def cal2NF(X):
# #     res = 0
# #     for x in X:
# #         res += x * x
# #     return res**0.5


# # def cal2N(data):
# #     res = 0
# #     for row in data:
# #         for x in row:
# #             res = res + x * x
# #     return res**0.5


# # def cal2NFF(X):
# #     return sum(multiply(X, X)) ** 0.5


# # X = array([[1, 2], [3, 4], [5, 5]])
# # print(cal2N(X))
# # print(cal2NF(X))
# # print(cal2NFF(X))
# v = array(range(20))
# print(v)
# indexes = random.randint(low=0, high=20, size=[10])
# print(indexes, end=" ")
# print("\n")
# print(v[indexes], end=" ")


# # test1与test2行列相同
# test1 = np.array([[1, 2], [3, 4]])
# test2 = np.array([[3, 3], [2, 2]])
# print(test1 * test2)
# # array([[3, 6], [6, 8]])
# print(np.dot(test1, test2))
# # array([[7, 7], [17, 17]])

a = array(
    [
        4,
        3,
        8,
        None,
        6,
        3,
        0,
        8,
        6,
        1,
        2,
        1,
        8,
        7,
        7,
        3,
        6,
        2,
        8,
        6,
        2,
        4,
        1,
        1,
        8,
        6,
        7,
        9,
        None,
        0,
    ]
)
print(a)

a = a.tolist()
print(a)
