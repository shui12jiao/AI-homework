from cmath import inf
from typing import Counter
from numpy import *
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris


def calcShannonEnt(dataSet):
    num = shape(dataSet)[0]
    type = unique(dataSet[:, -1])
    p = {}
    for i in type:
        print(dataSet[:, -1] == i)
        print(sum(dataSet[:, -1] == i))
        p[i] = sum(dataSet[:, -1] == i) / num
    shannonEnt = 0
    for i in type:
        shannonEnt -= p[i] * log2(p[i])
    return shannonEnt  # 返回经验熵


a = array([[1, 1], [2, 1], [1, 2], [2, 1], [1, 1]])
print(calcShannonEnt(a))
