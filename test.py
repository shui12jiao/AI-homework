from cmath import inf
from typing import Counter
from matplotlib.font_manager import FontProperties
from numpy import *
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris

# 函数说明：获取决策树叶子节点的数目  Parameters：myTree：决策树  Returns：numLeafs：决策树的叶子节点的数目
def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = next(iter(myTree))
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == "dict":
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs


# 函数说明:获取决策树的层数  Parameters: myTree:决策树  Returns: maxDepth:决策树的层数
def getTreeDepth(myTree):
    maxDepth = 0  # 初始化决策树深度
    firstStr = next(
        iter(myTree)
    )  # python3 中myTree.keys()返回的是dict_keys,不在是 list,所以不能使用myTree.keys()[0] 的方法获取结点属性，可以使用list(myTree.keys())[0]
    secondDict = myTree[firstStr]  # 获取下一个字典
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == "dict":  # 测试该结点是否为字典，如果不是字典，代表此结点为叶子结点
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth  # 更新层数
    return maxDepth


# 函数说明:绘制结点  Parameters: nodeTxt - 结点名 centerPt - 文本位置 parentPt - 标注的箭头位置 nodeType - 结点格式
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    arrow_args = dict(arrowstyle="<-")  # 定义箭头格式
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)  # 设置中文字体
    createPlot.ax1.annotate(
        nodeTxt,
        xy=parentPt,
        xycoords="axes fraction",  # 绘制结点
        xytext=centerPt,
        textcoords="axes fraction",
        va="center",
        ha="center",
        bbox=nodeType,
        arrowprops=arrow_args,
        font_properties=font,
    )


# 函数说明:标注有向边属性值  Parameters: cntrPt、parentPt - 用于计算标注位置 txtString - 标注的内容
def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]  # 计算标注位置
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)


# 函数说明:绘制决策树  Parameters: myTree - 决策树(字典) parentPt - 标注的内容 nodeTxt - 结点名
def plotTree(myTree, parentPt, nodeTxt):
    decisionNode = dict(boxstyle="sawtooth", fc="0.8")  # 设置结点格式
    leafNode = dict(boxstyle="round4", fc="0.8")  # 设置叶结点格式
    numLeafs = getNumLeafs(myTree)  # 获取决策树叶结点数目，决定了树的宽度
    depth = getTreeDepth(myTree)  # 获取决策树层数
    firstStr = next(iter(myTree))  # 下个字典
    cntrPt = (
        plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW,
        plotTree.yOff,
    )  # 中心位置
    plotMidText(cntrPt, parentPt, nodeTxt)  # 标注有向边属性值
    plotNode(firstStr, cntrPt, parentPt, decisionNode)  # 绘制结点
    secondDict = myTree[firstStr]  # 下一个字典，也就是继续绘制子结点
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD  # y 偏移
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == "dict":
            # 测试该结点是否为字典，如果不是字典，代表此结点为叶子结点
            plotTree(secondDict[key], cntrPt, str(key))  # 不是叶结点，递归调用继续绘制
        else:
            # 如果是叶结点，绘制叶结点，并标注有向边属性值
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD


# 函数说明:创建绘制面板  Parameters: inTree - 决策树(字典)
def createPlot(inTree):
    fig = plt.figure(1, facecolor="white")  # 创建 fig
    fig.clf()  # 清空 fig
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)  # 去掉 x、y 轴
    plotTree.totalW = float(getNumLeafs(inTree))  # 获取决策树叶结点数目
    plotTree.totalD = float(getTreeDepth(inTree))  # 获取决策树层数
    plotTree.xOff = -0.5 / plotTree.totalW
    plotTree.yOff = 1.0  # x 偏移
    plotTree(inTree, (0.5, 1.0), "")  # 绘制决策树
    plt.show()  # 显示绘制结果


tree = {
    "Flavanoids": {
        0.34: 3.0,
        0.47: 3.0,
        0.48: 3.0,
        0.49: 3.0,
        0.5: 3.0,
        0.51: 3.0,
        0.52: 3.0,
        0.55: 3.0,
        0.56: 3.0,
        0.57: 2.0,
        0.58: 3.0,
        0.6: 3.0,
        0.61: 3.0,
        0.63: 3.0,
        0.65: 3.0,
        0.66: 3.0,
        0.68: 3.0,
        0.69: 3.0,
        0.7: 3.0,
        0.75: 3.0,
        0.76: 3.0,
        0.78: 3.0,
        0.8: 3.0,
        0.83: 3.0,
        0.84: 3.0,
        0.92: 3.0,
        0.96: 3.0,
        0.99: 2.0,
        1.02: 2.0,
        1.09: {"Alcohol": {12.33: 2.0, 12.81: 3.0}},
        1.1: 3.0,
        1.2: 3.0,
        1.22: 3.0,
        1.25: {"Alcohol": {12.0: 2.0, 12.77: 2.0, 12.86: 3.0}},
        1.28: {"Alcohol": {12.21: 2.0, 13.11: 3.0}},
        1.3: 2.0,
        1.31: 3.0,
        1.32: 2.0,
        1.36: {"Alcohol": {12.6: 2.0, 12.79: 3.0}},
        1.39: 3.0,
        1.41: 2.0,
        1.46: 2.0,
        1.5: 2.0,
        1.57: {"Alcohol": {11.66: 2.0, 13.5: 3.0}},
        1.58: 2.0,
        1.59: 2.0,
        1.6: 2.0,
        1.61: 2.0,
        1.64: 2.0,
        1.69: 2.0,
        1.75: 2.0,
        1.76: 2.0,
        1.79: 2.0,
        1.84: 2.0,
        1.85: 2.0,
        1.92: 2.0,
        1.94: 2.0,
        2.0: 2.0,
        2.01: 2.0,
        2.03: 2.0,
        2.04: 2.0,
        2.09: 2.0,
        2.11: 2.0,
        2.13: 2.0,
        2.14: 2.0,
        2.17: 2.0,
        2.19: 1.0,
        2.21: 2.0,
        2.24: 2.0,
        2.25: 2.0,
        2.26: 2.0,
        2.27: 2.0,
        2.29: 2.0,
        2.33: 1.0,
        2.37: 1.0,
        2.41: 1.0,
        2.43: 1.0,
        2.45: 2.0,
        2.5: 2.0,
        2.51: 1.0,
        2.52: 1.0,
        2.53: {"Alcohol": {12.72: 2.0, 13.51: 1.0}},
        2.55: 2.0,
        2.58: 2.0,
        2.61: 1.0,
        2.63: 1.0,
        2.64: 1.0,
        2.65: {"Alcohol": {12.07: 2.0, 12.37: 2.0, 13.05: 2.0, 14.21: 1.0}},
        2.68: 1.0,
        2.69: 1.0,
        2.74: 1.0,
        2.76: 1.0,
        2.78: 1.0,
        2.79: {"Alcohol": {11.45: 2.0, 13.77: 1.0}},
        2.86: 2.0,
        2.88: 1.0,
        2.89: 2.0,
        2.9: 1.0,
        2.91: 1.0,
        2.92: {"Alcohol": {11.61: 2.0, 14.1: 1.0}},
        2.94: 1.0,
        2.97: 1.0,
        2.98: 1.0,
        2.99: {"Alcohol": {12.29: 2.0, 13.83: 1.0}},
        3.0: 1.0,
        3.03: {"Alcohol": {11.87: 2.0, 13.64: 1.0}},
        3.04: 1.0,
        3.06: 1.0,
        3.1: 2.0,
        3.14: 1.0,
        3.15: {"Alcohol": {12.43: 2.0, 13.86: 1.0}},
        3.17: 1.0,
        3.18: 2.0,
        3.19: 1.0,
        3.23: 1.0,
        3.24: 1.0,
        3.25: 1.0,
        3.27: 1.0,
        3.29: 1.0,
        3.32: 1.0,
        3.39: 1.0,
        3.4: 1.0,
        3.49: 1.0,
        3.54: 1.0,
        3.56: 1.0,
        3.64: 1.0,
        3.67: 1.0,
        3.69: 1.0,
        3.74: 1.0,
        3.75: 2.0,
        3.93: 1.0,
        5.08: 2.0,
    }
}
# createPlot(tree)

p = {1: 0.2, 2: 0.3, 3: 0.5}
print(max(p, key=p.get))
p = {1: 0.2, 2: 3, 3: 0.5}
print(max(p, key=p.get))
p = {1: 42, 2: 3, 3: 0.5}
print(max(p, key=p.get))
