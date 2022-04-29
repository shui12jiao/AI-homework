from cProfile import label
from calendar import c
import enum
from operator import index
from turtle import shape
import numpy as np
import struct
import math
import matplotlib.pyplot as plt
from sklearn.utils import compute_class_weight


acc0 = [0.9831, 0.9865, 0.9885, 0.9832, 0.9882]  # relu
loss0 = [  # relu
    0.051939789205789566,
    0.04128165915608406,
    0.034712061285972595,
    0.04705997183918953,
    0.03884277865290642,
]

acc1 = [0.9838, 0.9854, 0.9887, 0.9898, 0.9885]
loss1 = [
    0.052365124225616455,
    0.041685640811920166,
    0.03284215182065964,
    0.03471237048506737,
    0.03568664938211441,
]

x = [1, 2, 3, 4, 5]

plt.title("Average Loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.plot(x, loss1, c="red", label="More")
plt.plot(x, loss0, c="blue", label="Origin")
# plt.plot(x, loss3, c="green", label="Tanh")
# plt.plot(x, loss4, c="yellow", label="Sigmoid")
# plt.plot(x, loss5, c="black", label="Softmax")
plt.legend()
plt.show()


plt.title("Average Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.plot(x, acc1, c="red", label="More")
plt.plot(x, acc0, c="blue", label="Origin")
# plt.plot(x, acc3, c="green", label="Tanh")
# plt.plot(x, acc4, c="yellow", label="Sigmoid")
# plt.plot(x, acc5, c="black", label="Softmax")
plt.legend()
plt.show()
