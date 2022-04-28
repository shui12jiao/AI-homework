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


loss3 = [
    0.070694,
    0.083594,
    0.040785,
    0.037231,
    0.013847,
    0.058834,
    0.069244,
    0.047919,
    0.055576,
    0.021509,
    0.029240,
    0.070912,
    0.056319,
    0.105730,
    0.091816,
]


loss2 = [
    0.165976,
    0.074766,
    0.038812,
    0.087254,
    0.131406,
    0.155303,
    0.064809,
    0.072331,
    0.135663,
    0.051410,
    0.017742,
    0.136159,
    0.055516,
    0.075153,
    0.057956,
]

acc1 = [
    str(int(3712 *100/ 60000) ) + "%",
    str(int(7552 *100/ 60000) ) + "%",
    str(int(11392 *100/ 60000) ) + "%",
    str(int(15232 *100/ 60000) ) + "%",
    str(int(19072 *100/ 60000) ) + "%",
    str(int(22912 *100/ 60000) ) + "%",
    str(int(26752 *100/ 60000) ) + "%",
    str(int(30592 *100/ 60000) ) + "%",
    str(int(34432 *100/ 60000) ) + "%",
    str(int(38272 *100/ 60000) ) + "%",
    str(int(42112 *100/ 60000) ) + "%",
    str(int(45952 *100/ 60000) ) + "%",
    str(int(49792 *100/ 60000) ) + "%",
    str(int(53632 *100/ 60000) ) + "%",
    str(int(57472 *100/ 60000) ) + "%",
]

loss1 = [
    0.573305,
    0.420097,
    0.212601,
    0.220329,
    0.241694,
    0.064276,
    0.192987,
    0.219284,
    0.143011,
    0.202141,
    0.141700,
    0.079780,
    0.024670,
    0.134550,
    0.107223,
]

x = np.array(range(1, len(loss1) + 1))

plt.title("TrainSet Loss")
plt.ylabel("Loss")
plt.xlabel("Progress")
plt.plot(acc1, loss1, c="red", label="Epoch 1")
plt.plot(acc1, loss2, c="blue", label="Epoch 2")
plt.plot(acc1, loss3, c="green", label="Epoch 3")
plt.legend()
plt.show()


# plt.title("TrainSet Accuracy")
# plt.ylabel("Accuracy")
# plt.plot(x, acc1, c="red", label="Epoch 1")
# plt.plot(x, acc2, c="blue", label="Epoch 2")
# plt.plot(x, acc3, c="green", label="Epoch 3")
# plt.legend()
# plt.show()
