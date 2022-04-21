from cmath import inf
from typing import Counter
from matplotlib.font_manager import FontProperties
from numpy import *
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris


def cal2NF(X):
    res = 0
    for x in X:
        res += x * x
    return res**0.5


def cal2N(data):
    res = 0
    for row in data:
        for x in row:
            res = res + x * x
    return res**0.5


def cal2NFF(X):
    return sum(multiply(X, X)) ** 0.5


X = array([[1, 2], [3, 4], [5, 5]])
print(cal2N(X))
print(cal2NF(X))
print(cal2NFF(X))
