# 模拟银行间资产占比对违约银行数量的影响
import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']

def powlaw(k,gamma): # 返回一个幂律随机数
    # k是幂律常数，gamma是幂律分布的参数
    c = ((1 - random.uniform(0,1)) * (gamma - 1) / k) ** (1 - gamma)
    return c

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

def balanceSheet(G,k,gamma): # 返回银行信用拆借规模
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

def generateData_1(L,p,q,m): # 返回每个银行的资产负债表数据
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

def generateData_2(L,p,q,m,homo=False,*normal):
    # p ：银行间资产占总资产比例
    # q : 存款占总资产比例
    # m : 流动性资产占总资产比例
    IL = sum(L) # 银行间贷款
    IL = np.squeeze(np.asarray(IL))
    L = L.T
    IB = sum(L) # 银行间借款
    IB = np.squeeze(np.asarray(IB))
    TA = IL.sum() / p # 银行间总资产
    if homo == False:
        D = np.random.normal(normal[0],normal[1],N)
    else:
        D = homo * np.ones(N)
    BA = D / q
    M =  BA * m
    I = IB - IL + (1 - p - m) * TA / N # 投资

def generateData_3(BA): # 每种资产/负债的比例是随机的
    I = [];M = [];IL = [];IB = [];D = [];NW = []
    for i in range(N):
        proportion = np.random.uniform(0,1,6)
        proportion[:3] = proportion[:3] / proportion[:3].sum()
        proportion[3:] = proportion[3:] / proportion[3:].sum()
        I.append((proportion[0]))
        M.append((proportion[1]))
        IL.append((proportion[2]))
        IB.append((proportion[3]))
        D.append((proportion[4]))
        NW.append((proportion[5]))
    I = np.multiply(BA,np.array(I))
    M = np.multiply(BA,np.array(M))
    IL = np.multiply(BA,np.array(IL))
    IB = np.multiply(BA,np.array(IB))
    D = np.multiply(BA,np.array(D))
    NW = np.multiply(BA,np.array(NW))
    # # 构建资产负债表
    data = np.vstack((I,M,IL,IB,D,NW,BA)) # 按列合并矩阵
    data = data.T
    header = ['投资','流动性资产','银行间贷款','银行间借款','存款','净资产','总资产']
    bank = []
    for i in range(N):
        bank.append('bank' + str(i))
    Data = pd.DataFrame(data,columns=header,index=bank)
    return Data

def generateData_4(BA,p):
    TA = BA.sum()
    propotion1 = np.random.uniform(BA / TA / p / 2,BA / TA / p,(N))
    propotion2 = np.random.uniform(BA / TA / p / 2,BA / TA / p,(N))
    propotion1 = propotion1 / propotion1.sum()
    propotion2 = propotion2 / propotion2.sum()
    IL = TA * p * propotion1
    IB = TA * p * propotion2
    remain1 = BA - IL
    remain2 = BA - IB
    propotion3 = np.random.uniform(0,1,N)
    propotion4 = np.random.uniform(0,1,N)
    propotion5 = propotion3 + propotion4
    propotion3 = propotion3 / propotion5
    propotion4 = propotion4 / propotion5
    I = remain1 * propotion3
    M = remain1 * propotion3
    D = remain2 * propotion4
    NW = remain2 * propotion4
    # # 构建资产负债表
    data1 = np.vstack((I,M,IL,IB,D,NW,BA)) # 按列合并矩阵
    data1 = data1.T
    header1 = ['投资','流动性资产','银行间贷款','银行间借款','存款','净资产','总资产']
    bank = []
    for i in range(N):
        bank.append('bank' + str(i))
    Data1 = pd.DataFrame(data1,columns=header1,index=bank)
    # 构建资产负债表比例
    data2 = np.vstack((I/BA,M/BA,IL/BA,IB/BA,D/BA,NW/BA,BA/BA)) # 按列合并矩阵
    data2 = data2.T
    header2 = ['投资比例','流动性资产比例','银行间贷款比例','银行间借款比例','存款比例','净资产比例','总资产比例']
    bank = []
    for i in range(N):
        bank.append('bank' + str(i))
    Data2 = pd.DataFrame(data2,columns=header2,index=bank)
    return Data1,Data2

def constraint1(L):
    # L = L.reshape((N,N))
    IL_estimate = sum(L)
    IL_estimate = np.squeeze(np.asarray(IL_estimate))
    return sum(abs(IL - IL_estimate))

def constraint2(L):
    # L = L.reshape((N,N))
    L = L.T
    IB_estimate = sum(L)
    IB_estimate = np.squeeze(np.asarray(IB_estimate))
    return sum(abs(IB - IB_estimate))

def constraint3(L):
    # L = L.reshape((N,N))
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
        if constraint1(X) + constraint2(X) + constraint3(X) <= 1e-8:
            print('约束满足')
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
    E = (data['投资'] + data['流动性资产']) * (1 - shock) - data['流动性资产']
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
    # sum = 1
    if delta.sum() > 0.01:
        fundamental = set(np.where(ib - newib > 0)[0]) # 基础违约银行集合
    while delta.sum() > 0.01:
        ib = newib
        newib = -maxVector(-IB,-maxVector(np.dot(ib,pi) + E,np.zeros(N)))
        delta = abs(ib - newib)
    #     sum = sum + 1
    # print(sum)
    contagion = set(np.where(ib - newib > 0)[0])
    contagion = contagion.difference(fundamental) # 传染违约银行集合
    return newib,fundamental,contagion

global N # 银行个数
N = 200
M = 40

# 创建并绘制BA无标度网络
# G = nx.barabasi_albert_graph(N,1) 
# nx.draw(G,with_labels=True)
# plt.show()

# # 创建并绘制随机网络
# G = nx.erdos_renyi_graph(N,0.3)
# nx.draw(G,with_labels=True)
# plt.show()

# 得到信用关系矩阵1
# g = nx.to_numpy_array(G)
# 得到信用关系矩阵2
# r,R = relation(g,0.1,0.2)
# 得到借贷矩阵1
# L = balanceSheet(r,100,2)
leverage = [0.01]
avg_fundamental_number = []
avg_contagion_number = []
avg_default_number = []
for i in range(22):
    leverage.append((leverage[0] + 0.04 * i))

for p in leverage:
    avg_fundamental_number.append((0))
    avg_contagion_number.append((0))
    for i in range(M):
        data1,data2 = generateData_4(np.random.uniform(0,100,N) * 1000,p) # 用数据的第三种生成方式得到的资产负债表数据
        IB = np.array(data1['银行间借款'])
        IL = np.array(data1['银行间贷款'])
        ib = IB / IB.sum()
        il = IL / IL.sum()
        X = np.dot(ib.reshape((N,1)),il.reshape((1,N))) * IL.sum()
        X = X - np.diag(X.diagonal())
        L = RAS(X)
        shock1 = np.ones(N) * 0.2
        shock2 = abs(np.random.normal(0,0.4,(N)))
        ib,fundamental,contagion = clearVector(data1,L,shock2) # 得到清算向量与违约银行集合
        avg_fundamental_number[-1] = avg_fundamental_number[-1] + len(fundamental)
        avg_contagion_number[-1] = avg_contagion_number[-1] + len(contagion)
avg_fundamental_number = np.array(avg_fundamental_number) / M
avg_contagion_number = np.array(avg_contagion_number) / M
avg_default_number = avg_fundamental_number + avg_contagion_number
contagion_frequence = np.array(avg_contagion_number) / N # 计算违约传染概率
contagion_frequence[contagion_frequence < 0.05] = 0

fig1 = plt.figure(dpi=300)
plt.title('违约银行数量————杠杆')
plt.plot(leverage,avg_fundamental_number)
plt.plot(leverage,avg_contagion_number)
plt.plot(leverage,avg_default_number)
plt.xlabel('杠杆')
plt.ylabel('违约银行数量')
plt.legend(['基础违约银行数量','传染违约银行数量','总违约银行数量'])
fig1.savefig('C:/Users/Administrator/Desktop/毕业论文/答辩报告/违约银行数量————杠杆')

fig2 = plt.figure(dpi=300)
plt.title('传染概率————杠杆')
plt.plot(leverage,contagion_frequence)
plt.xlabel('杠杆')
plt.ylabel('传染概率')
fig1.savefig('C:/Users/Administrator/Desktop/毕业论文/答辩报告/传染概率————杠杆')

