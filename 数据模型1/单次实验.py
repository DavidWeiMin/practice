import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']

def powlaw(k,gamma): # 返回一个幂律随机数
    # k是幂律常数，gamma是幂律分布的参数
    c = ((1 - random.uniform(0,1)) * (gamma - 1) / k) ** (1 - gamma)
    return c

def relation(r,p1,p2): # 生成信用关系（包含借贷方向）
    # p1表示i为债务银行，j为债权银行的概率
    # p2表示j为债务银行，i为债权银行的概率
    # 1 - p1 - p2表示互为债权债务银行
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

def balanceSheet(G,k,gamma): # 返回银行信用拆借规模
    # 假设信用拆借规模服从幂律分布
    L = [] 
    for i in range(N ** 2):
        L.append(powlaw(k,gamma))
    L = np.array(L)
    L = L.reshape((N,N))                    # 产生一个NxN矩阵，矩阵数值均为幂律随机数
    L = np.multiply(L,G)                    # 与银行间信用关系矩阵对应相乘得到借贷矩阵
    return L

def maxVector(a,b): # 返回一个新向量，新向量由输入的两个向量对应位置的较大值构成
    c = []
    temp1 = a > b
    temp1.astype(int)
    temp2 = a < b
    temp2.astype(int)
    c = temp1 * a + temp2 * b
    return c

def generateData(L,p,q,m): # 返回每个银行的资产负债表数据
    # p : 银行间贷款占银行间总资产比例
    # q : 净资产占银行总资产比例
    # m : 流动性资产占银行总资产比例
    IL = sum(L)                             # 银行间贷款
    L = L.T
    IB = sum(L)                             # 银行间借款
    TA = IL.sum() / p                       # 银行间总资产
    I = IB - IL + (1 - p - m) * TA / N      # 投资
    M = m * (I + IL) /(1 - m)               # 流动性资产
    BA = I + M + IL                         # 银行总资产
    NW = BA * q                             # 净资产
    D = BA - IB - NW                        # 客户存款
    # 构建资产负债表
    data = np.vstack((I,M,IL,IB,D,NW,BA))   # 按列合并矩阵
    data = data.T
    header = ['投资','流动性资产','银行间贷款','银行间借款','存款','净资产','总资产']
    bank = []
    for i in range(N):
        bank.append('bank' + str(i))
    Data = pd.DataFrame(data,columns=header,index=bank)
    return Data

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
    pi = np.delete(pi,0,axis=0)                         #偿还比例矩阵
    ib = IB
    newib = -maxVector(-IB,-maxVector(np.dot(ib,pi) + E,np.zeros(N)))
    delta = abs(ib - newib)
    fundamental = set()
    if delta.sum() > 0.01:
        fundamental = set(np.where(ib - newib > 0)[0])  # 基础违约银行集合
    while delta.sum() > 0.01:
        ib = newib
        newib = -maxVector(-IB,-maxVector(np.dot(ib,pi) + E,np.zeros(N)))
        delta = abs(ib - newib)
    contagion = set(np.where(ib - newib > 0)[0])
    contagion = contagion.difference(fundamental)       # 传染违约银行集合
    return newib,fundamental,contagion

global N # 银行个数
N = 100
p = 0.3
q = 0.2
m = 0.3

G = nx.barabasi_albert_graph(N,1)                               # 创建无标度网络
average_clustering = nx.average_clustering(G)                   # 计算平均聚集度
average_degree_connectivity = nx.average_degree_connectivity(G) # 计算平均连接度
degree = G.degree()                                             # 得到每个节点的度
degree_histogram = nx.degree_histogram(G)                       # 网络度分布
g = nx.to_numpy_array(G)                                        # 得到信用关系矩阵(不含借贷方向)
r,R = relation(g,0.3,0.3)                                       # 得到信用关系矩阵(含借贷方向)
L = balanceSheet(r,100,2)                                       # 得到借贷规模矩阵
data = generateData(L,p,q,m)                                    # 得到资产负债表数据
shock = abs(np.random.normal(0,0.4,(N)))                        # 给出冲击
ib,fundamental,contagion = clearVector(data,L,shock)            # 计算清算向量与违约银行集合
LGD = 1 - ib / data['银行间借款']                                # 计算违约损失率
LGD_avg = 1 - ib.sum() / data['银行间借款'].sum()                # 计算平均违约损失率
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
fig1.savefig('D:/Documents/GitHub/practice/数据模型1/银行网络结构图（无向）')

# 银行网络结构图（有向）
fig2 = plt.figure(dpi=300)
plt.title('银行网络结构图（有向）')
nx.draw_shell(R,with_labels=True,font_size=6,node_size=80,node_color=color)
fig2.savefig('D:/Documents/GitHub/practice/数据模型1/银行网络结构图（有向）')

# 网络度分布
fig3 = plt.figure(dpi=300)
plt.title('网络度分布')
plt.bar(range(len(degree_histogram)),degree_histogram)
fig3.savefig('D:/Documents/GitHub/practice/数据模型1/度分布')
plt.xlabel('度')
plt.ylabel('银行个数')