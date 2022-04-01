from cmath import inf
from typing import Counter
from numpy import *
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris


INF = 0


def create_data():
    X = load_iris().data
    y = load_iris().target
    return X, y


lab = matrix(
    [
        [0],
        [2],
        [2],
        [0],
        [0],
        [2],
        [0],
        [2],
        [2],
        [1],
        [1],
        [2],
        [2],
        [0],
        [1],
        [1],
        [2],
        [1],
        [2],
        [1],
        [0],
        [0],
        [0],
        [2],
        [0],
        [1],
        [2],
        [2],
        [0],
        [0],
        [1],
        [0],
        [2],
        [1],
        [2],
        [2],
        [1],
        [2],
        [2],
        [1],
        [0],
        [1],
        [0],
        [1],
        [1],
        [0],
        [1],
        [0],
        [0],
        [2],
        [2],
        [2],
        [0],
        [0],
        [1],
        [0],
        [2],
        [0],
        [2],
        [2],
        [0],
        [2],
        [0],
        [1],
        [0],
        [1],
        [1],
        [0],
        [0],
        [1],
        [0],
        [1],
        [1],
        [0],
        [1],
        [1],
        [1],
        [1],
        [2],
        [0],
        [0],
        [2],
        [1],
        [2],
        [1],
        [2],
        [2],
        [1],
        [2],
        [0],
    ]
)

for i in range(10):
    if lab[i] * 1.2 < 1:
        print("good")
    else:
        print("no")
