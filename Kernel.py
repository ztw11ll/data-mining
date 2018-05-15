# -*- coding: utf-8 -*-

import numpy as np

result=[]
with open('iris.txt','r') as f:  #读取文件
    for line in f:
        line=list(map(str,line.split(','))) #逗号分割
        result.append(list(map(float,line[:4]))) #去除字母
D=np.array(result) #原数据
n=D.shape[0]#行数

K=np.empty((n,n),dtype = np.float) 
for i in range(0,n):   #计算每一个K元素的值
    for j in range(0,n):
        K[i][j]=np.dot(D[i,:],D[j,:]) 
K=K**2 #平方得到齐次二次核
I=np.eye(n)-np.full((n,n),1/n)
centeredK=np.dot(np.dot(I,K),I)#中心化
W=np.diag(np.diag(centeredK)**(-0.5))
normalizedK=np.dot(np.dot(W,centeredK),W)#归一化


def trf ():#二次齐次式转换
    for i in range(0,n):
        l2=[result[i][0]*result[i][1]*(2**0.5),result[i][0]*result[i][2]*(2**0.5),result[i][0]*result[i][3]*(2**0.5),result[i][1]*result[i][2]*(2**0.5),result[i][1]*result[i][3]*(2**0.5),result[i][2]*result[i][3]*(2**0.5)]
        result[i].extend(l2)
        for m in range(0,4):
            result[i][m]=result[i][m]**2
trf()
D2=np.array(result)

Dmean=D2.mean(axis=0)#中心化
Z=D2-np.ones((D2.shape[0],1),dtype=float)*Dmean

for x in range(0,n):#归一化
    Z[x]=Z[x]/(np.vdot(Z[x],Z[x])**0.5)
K2=np.zeros((n,n),dtype = np.float) 
for i in range(0,n):   #计算每一个K元素的值
    for j in range(0,n):
        K2[i][j]=np.vdot(Z[i,:],Z[j,:])
C=sum(sum(np.array(K2)-np.array(normalizedK)))#计算差值
print(float(C))