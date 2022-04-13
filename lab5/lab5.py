import operator
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
from numpy import *

# 函数说明：计算给定数据集的经验熵（香农熵）  Parameters：dataSet：数据集  Returns：shannonEnt：经验熵
def calcShannonEnt(dataSet):
    p = zeros(4)
    num = shape(dataSet)[0]
    for i in range(1,4):
        p[i] = sum(dataSet[-1]==i)/num
        # TODO
    for i in range(1,4):
        

    labels
    pass

    return shannonEnt  # 返回经验熵


# 函数说明：创建数据集  Returns：dataSet：数据集 labels：分类属性
def createDataSet(path):
    dataMat = []
    fr = open(path)
    for line in fr.readlines():
        lineArr = line.strip().split("\t")
        dataMat.append([lineArr[1:].float()])
    labels = [
        "age",
        "spectacle prescription",
        "astigmatic",
        "tear production rate",
    ]
    return array(dataSet), array(labels)


# 函数说明：按照给定特征划分数据集  Parameters：dataSet:待划分的数据集 axis：划分数据集的特征 value：需要返回的特征值  Returns：返回划分后的数据集
def splitDataSet(dataSet, axis, value):
    # 返回划分后的数据集
    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    # 这里调用 calcShannonEnt 函数来计算熵
    # 返回信息增益最大特征的索引值
    return bestFeature


# 函数说明：统计 classList 中出现次数最多的元素（类标签）  Parameters： classList：类标签列表  Returns： sortedClassCount[0][0]：出现次数最多的元素（类标签）
def majorityCnt(classList):
    classCount = {}
    # 统计 classList 中每个元素出现的次数
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
            classCount[vote] += 1
        # 根据字典的值降序排列
    sortedClassCount = sorted(
        classCount.items(), key=operator.itemgetter(1), reverse=True
    )
    return sortedClassCount[0][0]


# 函数说明：创建决策树  Parameters: dataSet：训练数据集 labels：分类属性标签 featLabels：存储选择的最优特征标签  Returns： myTree：决策树
def createTree(dataSet, labels, featLabels):
    return myTree


# 函数说明：获取决策树叶子节点的数目 Parameters：myTree：决策树  Returns： numLeafs：决策树的叶子节点的数目
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


# 函数说明:获取决策树的层数 Parameters: myTree:决策树  Returns: maxDepth:决策树的层数
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


# 函数说明:绘制结点 Parameters: nodeTxt - 结点名 centerPt - 文本位置 parentPt - 标注的箭头位置 nodeType - 结点格式
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
        FontProperties=font,
    )


# 函数说明:标注有向边属性值 Parameters: cntrPt、parentPt - 用于计算标注位置 txtString - 标注的内容
def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]  # 计算标注位置
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)


# 函数说明:绘制决策树 Parameters: myTree - 决策树(字典) parentPt - 标注的内容 nodeTxt - 结点名
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


if __name__ == "__main__":
    dataSet, labels = createDataSet()
    featLabels = []
    myTree = createTree(dataSet, labels, featLabels)
    print(myTree)
    createPlot(myTree)
