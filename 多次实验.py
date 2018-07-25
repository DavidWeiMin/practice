import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def powlaw(k,gamma): # 返回一个幂律随机数
    # k是幂律常数，gamma是幂律分布的参数
    c = ((1 - random.uniform(0,1)) * (gamma - 1) / k) ** (1 - gamma)
    return c

def relation(p1,p2,p3):
    R = nx.DiGraph()
    r = np.zeros((N,N))
    for i in range(N):
        for j in range(i):
            temp = random.uniform(0,1)
            if temp < p1:
                r[i,j] = 1
                R.add_edge(i,j)
            elif temp < p1 + p2:
                r[j,i] = 1
                R.add_edge(j,i)
            else:
                r[i,j] = 1
                r[j,i] = 1
                R.add_edge(i,j)
                R.add_edge(j,i)
    return r,R

def balanceSheet(G,k,gamma): # 返回银行借贷数据
    L = [] 
    for i in range(N ** 2):
        L.append(powlaw(k,gamma))
    L = np.array(L)
    L = L.reshape((N,N)) # 产生一个NxN矩阵，矩阵数值均为幂律随机数
    L = np.multiply(L,G) # 与银行间信用关系矩阵对应相乘得到借贷矩阵
    return L

def maxVector(a,b): # 返回一个新向量，新向量由输入的两个向量对应位置的较大值构成
    c = []
    temp1 = a > b
    temp1.astype(int)
    temp2 = a < b
    temp2.astype(int)
    c = temp1 * a + temp2 * b
    return c

def getData(L,p,q,m): # 返回每个银行的资产负债表数据
    # p : 银行间贷款占银行间总资产比例
    # q : 净资产占银行总资产比例
    # m : 流动性资产占银行总资产比例
    IL = sum(L) # 银行间贷款
    IL = np.squeeze(np.asarray(IL))
    L = L.T
    IB = sum(L) # 银行间借款
    IB = np.squeeze(np.asarray(IB))
    TA = IL.sum() / p # 银行间总资产
    I = IB - IL + (1 - p - m) * TA / N # 投资
    M = m * (I + IL) /(1 - m) # 流动性资产
    BA = I + M + IL # 银行总资产
    NW = BA * q # 净资产
    D = BA - IB - NW # 客户存款
    L = L.T
    # # 构建资产负债表
    data = np.vstack((I,M,IL,IB,D,NW,BA)) # 按列合并矩阵
    data = data.T
    header = ['投资','流动性资产','银行间贷款','银行间借款','存款','净资产','总资产']
    bank = []
    for i in range(N):
        bank.append('bank' + str(i))
    Data = pd.DataFrame(data,columns=header,index=bank)
    return Data

def clearVector(data,L): # 返回清算向量与违约银行
    IB = data['银行间借款']
    IB = IB.as_matrix()
    E = (data['投资'] + data['流动性资产']) * 0.25 - data['流动性资产']
    E = E.as_matrix()
    pi = np.zeros(N)
    for index,value in enumerate(L):
        pi = np.row_stack((pi,value/IB[index]))
    pi = np.delete(pi,0,axis=0)
    ib = IB
    newib = -maxVector(-IB,-maxVector(np.dot(ib,pi) + E,np.zeros(N)))
    delta = abs(ib - newib)
    fundamental = set()
    if delta.sum() > 0.01:
        fundamental = set(np.where(ib - newib > 0)[0]) # 基础违约银行集合
    count = 1
    while delta.sum() > 0.01:
        ib = newib
        newib = -maxVector(-IB,-maxVector(np.dot(ib,pi) + E,np.zeros(N)))
        delta = abs(ib - newib)
        count = count + 1
    print(count)
    contagion = set(np.where(ib - newib > 0)[0])
    contagion = contagion.difference(fundamental) # 传染违约银行集合
    return newib,fundamental,contagion

global N # 银行个数
N = 200
p = 0.5
m = 0.1

# 创建并绘制BA无标度网络
G = nx.barabasi_albert_graph(N,1) 
nx.draw(G,with_labels=True)
plt.show()

g = nx.to_numpy_array(G) # 得到信用关系矩阵
r,R = relation(0.1,0.2,0.7)
L = balanceSheet(r,100,2) # 得到借贷矩阵
leverage = [1 / 9]
fundamental_number = []
contagion_number = []
default_number = []
for i in range(40):
    leverage.append((leverage[0] + 0.1 * i))
for i in leverage:
    q = i /(1 + i)
    data = getData(L,p,q,m) # 得到资产负债表数据
    ib,fundamental,contagion = clearVector(data,L) # 得到清算向量与违约银行集合
    fundamental_number.append((len(fundamental)))
    contagion_number.append((len(contagion)))
    default_number.append((len(fundamental) + len(contagion)))
plt.plot(leverage,fundamental_number,contagion_number)
plt.legend(['fundamental default','contagion default'])
plt.show()
