import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']

def maxVector(a,b): # 返回一个新向量，新向量由输入的两个向量对应位置的较大值构成
    c = []
    temp1 = a > b
    temp1.astype(int)
    temp2 = a < b
    temp2.astype(int)
    c = temp1 * a + temp2 * b
    return c

def generateData(BA,p):
    TA = BA.sum()
    IL = np.random.uniform(p * BA.sum() /N,BA,(N))
    IL= IL / IL.sum() * p * BA.sum()
    IB = np.random.uniform(p * BA.sum() /N,BA,(N))
    IB = IB / IB.sum() * p * BA.sum()
    remain1 = BA - IL
    remain2 = BA - IB
    propotion3 = np.random.uniform(0,1,(N,2))
    propotion4 = np.random.uniform(0,1,(N,2))
    propotion5 = propotion3 + propotion4
    propotion3 = propotion3 / propotion5
    propotion4 = propotion4 / propotion5
    I = remain1 * propotion3[:,0]
    M = remain1 * propotion4[:,0]
    D = remain2 * propotion3[:,1]
    NW = remain2 * propotion4[:,1]
    # 构建资产负债表
    data = np.vstack((I,M,IL,IB,D,NW,BA)) # 按列合并矩阵
    data = data.T
    header = ['投资','流动性资产','银行间贷款','银行间借款','存款','净资产','总资产']
    bank = []
    for i in range(N):
        bank.append('银行' + str(i))
    Data = pd.DataFrame(data,columns=header,index=bank)
    return Data

def constraint1(L):
    IL_estimate = sum(L)
    return sum(abs(IL - IL_estimate))

def constraint2(L):
    L = L.T
    IB_estimate = sum(L)
    return sum(abs(IB - IB_estimate))

def constraint3(L):
    return sum(abs(L.diagonal()))

def RAS(X):
    for i in range(1000):
        temp = X.copy()
        row = IB / sum(X.T)
        row = np.array([row] * N)
        X = X * row.T
        column = IL / sum(X)
        column = np.array([column] * N)
        X = X * column
        if constraint1(X) + constraint2(X) + constraint3(X) <= 1e-10:
            print('status : success')
            print('iteration : ',i)
            break
        if sum(sum(abs(temp - X))) < 1e-10:
            print('function out of tolerance')
            print('iteration : ',i)
            break
        if i == 999:
            print('out of max iteration')
    return X

def clearVector(data,L,shock): # 返回清算向量与违约银行
    IB = data['银行间借款']
    IB = IB.as_matrix()
    E = (data['投资'] + data['流动性资产']) * (1 - shock) - data['存款']
    E = E.as_matrix()
    pi = np.zeros(N)
    for index,value in enumerate(L):
        if IB[index] == 0:
            pi = np.row_stack((pi,np.zeros(N)))
        else:
            pi = np.row_stack((pi,value/IB[index]))
    pi = np.delete(pi,0,axis=0)
    ib = IB
    newib = -maxVector(-IB,-maxVector(np.dot(ib,pi) + E,np.zeros(N)))
    delta = abs(ib - newib)
    fundamental = set()
    if delta.sum() > 0.01:
        fundamental = set(np.where(ib - newib > 0)[0]) # 基础违约银行集合
    while delta.sum() > 0.01:
        ib = newib
        newib = -maxVector(-IB,-maxVector(np.dot(ib,pi) + E,np.zeros(N)))
        delta = abs(ib - newib)
    contagion = set(np.where(ib - newib > 0)[0])
    contagion = contagion.difference(fundamental) # 传染违约银行集合
    return newib,fundamental,contagion

def VaR(distribution,alpha):
    distribution = np.array(distribution)
    return np.percentile(distribution,alpha)

def ES(distribution,alpha):
    percentile = np.linspace(alpha,99.99,1000)
    ES = 0
    for i in percentile:
        ES = ES + VaR(distribution,i) / 1000
    return ES 

global N # 银行个数
N = 50

BA = abs(np.random.uniform(1,2,N)) * 1000
p_max = N * min(BA)/BA.sum()
p = p_max * 0.5
shock = abs(np.random.normal(0,0.4,(N)))                    # 给出冲击
fundamental_number = []                                     #初始化基础违约矩阵
contagion_number = []                                       #初始化传染违约矩阵
default_number = []
LGD_avg = []
for i in range(100):
    data = generateData(BA,p) # 用数据的第三种生成方式得到的资产负债表数据
    # 利用数据1进行计算
    IB = np.array(data['银行间借款'])
    IL = np.array(data['银行间贷款'])
    ib = IB / IB.sum()
    il = IL / IL.sum()
    X = np.dot(ib.reshape((N,1)),il.reshape((1,N))) * IL.sum()
    X = X - np.diag(X.diagonal())
    L = RAS(X)
    ib,fundamental,contagion = clearVector(data,L,shock)    # 得到清算向量与违约银行集合
    LGD_avg.append((1 - ib.sum() / data['银行间借款'].sum()))
    fundamental_number.append((len(fundamental)))
    contagion_number.append((len(contagion)))
    default_number.append((len(fundamental) + len(contagion)))
    fundamental_number.append((len(fundamental)))           # 添加基础违约银行数量
    contagion_number.append((len(contagion)))               # 添加传染违约银行数量
default_number = np.array(fundamental_number) + np.array(contagion_number)

fig1 = plt.figure(dpi=300)
plt.hist(fundamental_number,bins=max(fundamental_number) - min(fundamental_number))
plt.xlabel('破产银行数量')
plt.ylabel('频数')
fig1.savefig('D:/Documents/GitHub/practice/数据模型3/完全网络结构：基础违约分布')

fig2 = plt.figure(dpi=300)
plt.hist(fundamental_number,bins=max(contagion_number) - min(contagion_number))
plt.xlabel('破产银行数量')
plt.ylabel('频数')
fig2.savefig('D:/Documents/GitHub/practice/数据模型3/完全网络结构：传染违约分布')

fig3 = plt.figure(dpi=300)
plt.hist(fundamental_number,bins=max(default_number) - min(default_number))
plt.xlabel('破产银行数量')
plt.ylabel('频数')
fig3.savefig('D:/Documents/GitHub/practice/数据模型3/完全网络结构：总违约分布')

fig4 = plt.figure(dpi=300)
plt.hist(LGD_avg)
plt.xlabel('平均违约损失率')
plt.ylabel('频数')
fig4.savefig('D:/Documents/GitHub/practice/数据模型3/完全网络结构：平均违约损失率分布')

statistics = pd.DataFrame(index=['mean','median','VaR','ES'],columns=['基础违约','传染违约'])
statistics['基础违约']=[np.mean(fundamental_number),np.median(fundamental_number),VaR(fundamental_number,95),ES(fundamental_number,95)]
statistics['传染违约']=[np.mean(contagion_number),np.median(contagion_number),VaR(contagion_number,95),ES(contagion_number,95)]
statistics['总违约']=[np.mean(default_number),np.median(default_number),VaR(default_number,95),ES(default_number,95)]
print(statistics)