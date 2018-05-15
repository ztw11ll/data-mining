# -*- coding: utf-8 -*-

import math   # 导入 math 模块
import operator 
from collections import Counter
def calcShannonEnt(dataSet):
    '''
    计算香农熵
    :param dataSet:数据集
    :return: 计算结果
    '''
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet: # 遍历每个实例，统计标签的频数
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys(): 
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * math.log(prob,2) # 以2为底的对数
    return shannonEnt
def splitDataSet(dataSet, axis, value):
    '''
    按照给定特征划分数据集
    :param dataSet:待划分的数据集
    :param axis:划分数据集的特征
    :param value: 需要返回的特征的值
    :return: 划分结果列表
    '''
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]     #chop out axis used for splitting
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet
def calcConditionalEntropy(dataSet, i, featList, uniqueVals):
    '''
    计算X_i给定的条件下，Y的条件熵
    :param dataSet:数据集
    :param i:维度i
    :param featList: 数据集特征列表
    :param uniqueVals: 数据集特征集合
    :return: 条件熵
    '''
    conditionEnt = 0.0
    for value in uniqueVals:
        subDataSet = splitDataSet(dataSet, i, value)
        prob = len(subDataSet) / float(len(dataSet))  # 极大似然估计概率
        conditionEnt += prob * calcShannonEnt(subDataSet)  # 条件熵的计算
    return conditionEnt
def calcInformationGain(dataSet, baseEntropy, i):
    '''
    计算信息增益
    :param dataSet:数据集
    :param baseEntropy:数据集的信息熵
    :param i: 特征维度i
    :return: 特征i对数据集的信息增益g(D|X_i)
    '''
    featList = [example[i] for example in dataSet]  # 第i维特征列表
    uniqueVals = set(featList)  # 转换成集合
    newEntropy = calcConditionalEntropy(dataSet, i, featList, uniqueVals)
    infoGain = baseEntropy - newEntropy  # 信息增益，就yes熵的减少，也就yes不确定性的减少
    return infoGain
def chooseBestFeatureToSplitByID3(dataSet):
    '''
            选择最好的数据集划分方式
    :param dataSet:数据集
    :return: 划分结果
    '''
    numFeatures = len(dataSet[0]) - 1  # 最后一列yes分类标签，不属于特征向量
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):  # 遍历所有特征
        infoGain = calcInformationGain(dataSet, baseEntropy, i)     # 计算信息增益
        if (infoGain > bestInfoGain):  # 选择最大的信息增益
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature  # 返回最优特征对应的维度

def majorityCnt(classList):
    '''
    采用多数表决的方法决定叶结点的分类
    :param: 所有的类标签列表
    :return: 出现次数最多的类
    '''
    classCount={}
    for vote in classList:                  # 统计所有类标签的频数
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key= operator.itemgetter(1), reverse=True) # 降序排序
    return sortedClassCount[0][0]

def createTree(dataSet,labels): #创建决策树 ，dataSet:训练数据集，labels:属性

    classList = [example[-1] for example in dataSet]
    data = Counter(classList)
    purity=data.most_common(1)[0][1]/len(classList) #计算纯度
    if purity >= 0.95: 
        return "type:"+data.most_common(1)[0][0]+" purity:"+str(purity)+" size:"+str(len(dataSet))            # 第一个递归结束条件：纯度高于95%
    if len(dataSet) <= 5:        
        return "type:"+data.most_common(1)[0][0]+" purity:"+str(purity)+" size:"+str(len(dataSet))  # 第二个递归结束条件：节点数量小于5
    bestFeat = chooseBestFeatureToSplitByID3(dataSet)   # 最优划分特征
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}         # 使用字典类型储存树的信息
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)        #删除重复数据
    for value in uniqueVals:            #创建子树
        subLabels = labels[:]           # 复制所有类标签，保证每次递归调用时不改变原始列表的内容
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),subLabels)
    return myTree
result=[]
with open('iris.txt','r') as f:  #读取文件
    for line in f:
        line=list(map(str,line.split(','))) #逗号分割
        result.append(list(map(str,line))) #去除字母
labels = ['attribute1', 'attribute2', 'attribute3', 'attribute4']
myTree = createTree(result, labels)
print(myTree)
#createPlot(myTree)
out=open('out.txt','w')
print(myTree,file=out)
out.close()
