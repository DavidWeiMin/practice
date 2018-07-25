import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']

def relation(r,p1,p2): # 生成信用关系
    R = nx.DiGraph()
    for i in range(N):
        for j in range(i):
            if r[i,j] == 1:
                temp = random.uniform(0,1)
                if temp < p1:
                    r[j,i] = 0
                    R.add_edge(i,j)
                elif temp < p1 + p2:
                    r[i,j] = 0
                    R.add_edge(j,i)
                else:
                    R.add_edge(i,j)
                    R.add_edge(j,i)
    return r,R

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

def entropy2(L):
    y = 0
    for i in range(N):
        for j in range(i):
            y = y + math.log(L[i,j] / X[i,j]) * L[i,j]
            y = y + math.log(L[j,i] / X[j,i]) * L[j,i]
    return y

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

global N # 银行个数
N = 50

BA = abs(np.random.uniform(1,2,N)) * 1000
p_max = N * min(BA)/BA.sum()
p = p_max * 0.5
data = generateData(BA,p) # 用数据的第三种生成方式得到的资产负债表数据
# 利用数据1进行计算
IB = np.array(data['银行间借款'])
IL = np.array(data['银行间贷款'])
ib = IB / IB.sum()
il = IL / IL.sum()
X = np.dot(ib.reshape((N,1)),il.reshape((1,N))) * IL.sum()
X = X - np.diag(X.diagonal())
L = RAS(X)
print(entropy2(L))
shock1 = np.ones(N) * 0.2
shock2 = abs(np.random.normal(0,0.4,(N)))
ib,fundamental,contagion = clearVector(data,L,shock1) # 得到清算向量与违约银行集合
# 计算违约损失率
LGD = 1 - ib / data['银行间借款']
LGD_avg = 1 - ib.sum() / data['银行间借款'].sum()
print('基础违约银行数量：',len(fundamental))
print('传染违约银行数量：',len(contagion))

# 银行网络结构图（无向）
color = []
for i in range(N):
    if i in fundamental:
        color.append(('g'))
    elif i in contagion:
        color.append(('b'))
    else:
        color.append(('r'))
fig1 = plt.figure(dpi=300)
plt.title('银行网络结构图（无向）')
nx.draw(G,with_labels=True,font_size=6,node_size=80,node_color=color,label=['银行']*N)
fig1.savefig('C:/Users/Administrator/Desktop/毕业论文/答辩报告/银行网络结构图（无向）')

# 银行网络结构图（有向）
fig2 = plt.figure(dpi=300)
plt.title('银行网络结构图（有向）')
nx.draw_shell(R,with_labels=True,font_size=6,node_size=80,node_color=color)
fig2.savefig('C:/Users/Administrator/Desktop/毕业论文/答辩报告/银行网络结构图（有向）')

# 网络度分布
fig3 = plt.figure(dpi=300)
plt.title('网络度分布')
plt.bar(range(len(degree_histogram)),degree_histogram)
fig3.savefig('C:/Users/Administrator/Desktop/毕业论文/答辩报告/度分布')
plt.xlabel('度')
plt.ylabel('银行个数')