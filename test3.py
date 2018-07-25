import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def powlaw(k,gamma):
    c = ((1 - random.uniform(0,1)) * (gamma - 1) / k) ** (1 - gamma)
    return c

def balanceSheet(G,k,gamma):
    L = []
    for i in range(N ** 2):
        L.append(powlaw(k,gamma))
    L = np.array(L)
    L = L.reshape((N,N))
    L = np.multiply(L,G)
    return L

def minVector(a,b):
    c = []
    temp1 = a < b
    temp1.astype(int)
    temp2 = a > b
    temp2.astype(int)
    c = temp1 * a + temp2 * b
    return c


def clearVector(L,p,q,m):
    IL = sum(L)
    IL = np.squeeze(np.asarray(IL))
    L = L.T
    IB = sum(L)
    IB = np.squeeze(np.asarray(IB))
    TA = IL.sum() / p # 银行间贷款占银行间总资产比例
    I = IB - IL + (1 - p - m) * TA / N
    M = m * (I + IL) /(1 - m) # 流动性资产
    BA = I + M + IL # 某个银行总资产
    NW = BA * q # 净资产比例
    D = BA - IB - NW # 客户存款
    EA = I + M
    EA = EA / 2 # 外部资产损失一半
    E = EA - M # 其实就是投资 I
    L = L.T
    # # 构建资产负债表
    data = np.vstack((I,M,IL,IB,D,NW)) # 按列合并矩阵
    data = data.T
    header = ['投资','流动性资产','银行间贷款','银行间借款','存款','净资产']
    # header = ['投资','流动性资产','银行间贷款','银行间借款','存款','净资产','基础违约','传染违约']
    bank = []
    for i in range(N):
        bank.append('bank' + str(i))
    collect = pd.DataFrame(data,columns=header,index=bank)
    # collect = pd.DataFrame(columns=header,index=bank)
    # collect[['投资','流动性资产','银行间贷款','银行间借款','存款','净资产']]=data
    ##############################
    pi = np.zeros(N)
    for index,value in enumerate(L):
        pi = np.row_stack((pi,value/IB[index]))
    pi = np.delete(pi,0,axis=0)
    ib = IB
    newib = minVector(IB,np.squeeze(np.asarray(np.dot(ib,pi) + E)))
    newib[newib < 0] = 0
    delta = abs(newib - ib)
    while delta.sum() > 0.01:
        ib = newib
        newib = minVector(IB,np.squeeze(np.asarray(np.dot(ib,pi) + E)))
        newib[newib < 0] = 0
        delta = abs(newib - ib)
    # index = np.array(np.where(IB - newib > 0))
    # delta = IB - np.squeeze(np.asarray(np.dot(IB,pi) + E))
    # for i in index:
    #     if delta[i] > 0:
    #         collect.ix[i,['基础违约']] = True
    #     else:
    #         collect.ix[i,['传染违约']] = True 
    # collect['基础违约'].fillna = False
    # collect['传染违约'].fillna = False
    
    return IB-newib,collect
    
global N
N = 20
# BA无标度网络
G = nx.barabasi_albert_graph(N,3)
nx.draw_shell(G,with_labels=True)
plt.show()
g = nx.to_numpy_matrix(G)
L = balanceSheet(g,100,2)
ib,collect = clearVector(L,0.08,0.3,0.05)
print(ib)
print(collect)

# 1、自动化输出基础、传染违约银行与数量
# 2、进行蒙特卡洛模拟得到系统性风险分布
# 3、试验不同参数对风险的影响