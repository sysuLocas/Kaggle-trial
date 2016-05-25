#!/usr/bin/python
#-*-coding:utf-8-*-

from numpy import *
import operator
import csv
def toInt(array):
    array=mat(array)
    m,n=shape(array)
    newArray=zeros((m,n))
    for i in xrange(m):
        for j in xrange(n):
                newArray[i,j]=int(array[i,j])
    return newArray
    
def nomalizing(array):
    m,n=shape(array)
    for i in xrange(m):
        for j in xrange(n):
            if array[i,j]!=0:
                array[i,j]=1
    return array
    
def loadTrainData():
    l=[]
    with open('train.csv') as file:
         lines=csv.reader(file)
         for line in lines:
             l.append(line) #42001*785
    l.remove(l[0])
    l=array(l) #使用array函数从常规的Python列表和元组创造数组
    label=l[:,0]
    data=l[:,1:]
    return nomalizing(toInt(data)),toInt(label)  #label 1*42000  data 42000*784
    #return data,label
    
def loadTestData():
    l=[]
    with open('test.csv') as file:
         lines=csv.reader(file)
         for line in lines:
             l.append(line)
     #28001*784
    l.remove(l[0])
    data=array(l)
    return nomalizing(toInt(data))  #  data 28000*784

def loadTestResult():
    l=[]
    with open('knn_benchmark.csv') as file:
         lines=csv.reader(file)
         for line in lines:
             l.append(line)
     #28001*2
    l.remove(l[0])
    label=array(l)
    return toInt(label[:,1])  #  label 28000*1

#dataSet:m*n   labels:m*1  inX:1*n
def classify(inX, dataSet, labels, k):#inX是要被测试的向量
    inX=mat(inX)
    dataSet=mat(dataSet)
    labels=mat(labels)
    dataSetSize = dataSet.shape[0]#训练数据的行数                  
    
    #下面这一段是计算差的平方和再开根
    diffMat = tile(inX, (dataSetSize,1)) - dataSet#tile(inX,(n,1))函数，将inX做行重复拓展n次，得到差值矩阵  http://jingyan.baidu.com/article/219f4bf7da4d8dde442d389e.html 
    sqDiffMat = array(diffMat)**2
    sqDistances = sqDiffMat.sum(axis=1)#array.sum(axis=1)按行累加，axis=0为按列累加                  
    distances = sqDistances**0.5
    #
    
    sortedDistIndicies = distances.argsort()#记录排序后的索引值，有了索引值可以找到最小的前K个训练样本的《索引位置》            
    classCount={}#创建一个空字典                                      
    for i in range(k):
        voteIlabel = labels[0,sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1 #get(key,x)从字典中获取key对应的value，没有key的话返回0
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True) #sorted()函数，按照第二个元素即value的次序逆向（reverse=True）排序
    return sortedClassCount[0][0]

def saveResult(result):
    with open('result.csv','wb') as myFile:    
        myWriter=csv.writer(myFile)
        for i in result:
            tmp=[]
            tmp.append(i)
            myWriter.writerow(tmp)
        

def handwritingClassTest():
    trainData,trainLabel=loadTrainData()#加载训练集数据以及其对应的标签
    #print(trainLabel[0,3])
    testData=loadTestData()#加载测试集数据
    testLabel=loadTestResult()#加载测试集的对应标签 open('knn_benchmark.csv')
    m,n=shape(testData)#测试集的行数、列数
    errorCount=0
    resultList=[]#列表初始化
    for i in range(m):#对测试集每一行做循环i从0开始的
         classifierResult = classify(testData[i],trainData[0:1000,:],trainLabel[0,0:1000],5)#测试集每一行执行一次，带参有：训练数据集及其对应的标签、k
         resultList.append(classifierResult)#结果加入到结果列表中
         print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, testLabel[0,i])
         if (classifierResult != testLabel[0,i]): errorCount += 1.0
    print "\nthe total number of errors is: %d" % errorCount
    print "\nthe total error rate is: %f" % (errorCount/float(m))
    saveResult(resultList)
