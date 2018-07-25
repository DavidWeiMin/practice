import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from pylab import mpl
import scipy.optimize as op
mpl.rcParams['font.sans-serif'] = ['SimHei']

def powlaw(k,gamma): # 返回一个幂律随机数
    # k是幂律常数，gamma是幂律分布的参数,gamma越大，异质性越小
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

def generateData(IL,IB,p):
    cons = ({'type':'ineq','fun':lambda BA:BA - 1.11 * IL},{'type':'ineq','fun':lambda BA:BA - 1.11 * IB},{'type':'eq','fun':lambda x: x.sum() - IL.sum() / p})
    res = op.minimize(lambda BA: -sum((BA - IL.sum() / p / N )** 2) / N,np.ones(N) * IL.sum() / p / N,constraints=cons)
    BA = res.x
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
    # # 构建资产负债表
    data = np.vstack((I,M,IL,IB,D,NW,BA)) # 按列合并矩阵
    data = data.T
    header = ['投资','流动性资产','银行间贷款','银行间借款','存款','净资产','总资产']
    bank = []
    for i in range(N):
        bank.append('银行' + str(i))
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
N = 20

G = nx.barabasi_albert_graph(N,1)                               # 创建无标度网络
average_clustering = nx.average_clustering(G)                   # 计算平均聚集度
average_degree_connectivity = nx.average_degree_connectivity(G) # 计算平均连接度
degree = G.degree()                                             # 得到每个节点的度
degree_histogram = nx.degree_histogram(G)                       # 网络度分布
g = nx.to_numpy_array(G)                                        # 得到信用关系矩阵(不含借贷方向)
r,R = relation(g,0,0)                                           # 得到信用关系矩阵(含借贷方向)
L = balanceSheet(r,100,2)                                       # 得到借贷规模矩阵
IL = sum(L) # 银行间贷款
L = L.T
IB = sum(L) # 银行间借款
m1 = 100
leverage = np.linspace(0.01,0.45,m1)
m2 = 4
loss = np.linspace(0.1,0.8,m2)
Mean_fundamental = [];Mean_contagion = [];Mean_default = [];Mean_LGD_avg = []
Median_fundamental = [];Median_contagion = [];Median_default = [];Median_LGD_avg = []
VaR_fundamental = [];VaR_contagion = [];VaR_default = [];VaR_LGD_avg = []
ES_fundamental = [];ES_contagion = [];ES_default = [];ES_LGD_avg = []
for i in loss:
    shock = np.ones(N) * i
    for p in leverage:
        fundamental_number = []
        contagion_number = []
        default_number = []
        LGD_avg = []
        for j in range(80):
            data = generateData(IL,IB,p) # 得到资产负债表数据
            temp = np.array(data < 0)
            if True in temp:
                print('error')
            ib,fundamental,contagion = clearVector(data,L,shock) # 得到清算向量与违约银行集合
            LGD_avg.append((1 - ib.sum() / data['银行间借款'].sum()))
            fundamental_number.append((len(fundamental)))
            contagion_number.append((len(contagion)))
        Mean_fundamental.append((np.mean(fundamental_number)))
        Mean_contagion.append((np.mean(contagion_number)))
        Mean_LGD_avg.append((np.mean(LGD_avg)))
        Median_fundamental.append((np.median(fundamental_number)))
        Median_contagion.append((np.median(contagion_number)))
        Median_LGD_avg.append((np.median(LGD_avg)))
        VaR_fundamental.append((VaR(fundamental_number,95)))
        VaR_contagion.append((VaR(contagion_number,95)))
        VaR_LGD_avg.append((VaR(LGD_avg,95)))
        ES_fundamental.append((ES(fundamental_number,95)))
        ES_contagion.append((ES(contagion_number,95)))
        ES_LGD_avg.append((ES(LGD_avg,95)))
Mean_fundamental = np.array(Mean_fundamental)
Mean_contagion = np.array(Mean_contagion)
Mean_LGD_avg = np.array(Mean_LGD_avg)
Median_fundamental = np.array(Median_fundamental)
Median_contagion = np.array(Median_contagion)
Median_LGD_avg = np.array(Median_LGD_avg)
VaR_fundamental = np.array(VaR_fundamental) 
VaR_contagion = np.array(VaR_contagion)
VaR_LGD_avg = np.array(VaR_LGD_avg)
ES_fundamental = np.array(ES_fundamental)
ES_contagion = np.array(ES_contagion)
ES_LGD_avg = np.array(ES_LGD_avg)
# contagion_frequence = contagion_number / N # 计算违约传染概率
# contagion_frequence[contagion_frequence < 0.05] = 0
Mean_fundamental = Mean_fundamental.reshape((m2,m1))
Mean_contagion = Mean_contagion.reshape((m2,m1))
Mean_LGD_avg = Mean_LGD_avg.reshape((m2,m1))
Median_fundamental = Median_fundamental.reshape((m2,m1))
Median_contagion = Median_contagion.reshape((m2,m1))
Median_LGD_avg = Median_LGD_avg.reshape((m2,m1))
VaR_fundamental = VaR_fundamental.reshape((m2,m1))
VaR_contagion = VaR_contagion.reshape((m2,m1))
VaR_LGD_avg = VaR_LGD_avg.reshape((m2,m1))
ES_fundamental = ES_fundamental.reshape((m2,m1))
ES_contagion = ES_contagion.reshape((m2,m1))
ES_LGD_avg = ES_LGD_avg.reshape((m2,m1))
Mean_default = Mean_fundamental + Mean_contagion
Median_default = Median_fundamental + Median_contagion
VaR_default = VaR_fundamental + VaR_contagion
ES_default = ES_fundamental + ES_contagion

fig1 = plt.figure(dpi=300)
legend = ['外部资产损失=%f' % round(i,2) for i in loss]
for i in range(m2):
    plt.plot(leverage,Mean_fundamental[i][:])
plt.title('基础违约银行数量平均数————杠杆')
plt.xlabel('杠杆')
plt.ylabel('基础违约银行数量平均数')
plt.legend(legend)
fig1.savefig('D:/Documents/GitHub/practice/数据模型2/改变外部资产损失率：1-基础违约银行数量平均数————杠杆')

fig2 = plt.figure(dpi=300)
legend = ['外部资产损失=%f' % round(i,2) for i in loss]
for i in range(m2):
    plt.plot(leverage,Median_fundamental[i][:])
plt.title('基础违约银行数量中位数————杠杆')
plt.xlabel('杠杆')
plt.ylabel('基础违约银行数量中位数')
plt.legend(legend)
fig2.savefig('D:/Documents/GitHub/practice/数据模型2/改变外部资产损失率：2-基础违约银行数量中位数————杠杆')

fig3 = plt.figure(dpi=300)
legend = ['外部资产损失=%f' % round(i,2) for i in loss]
for i in range(m2):
    plt.plot(leverage,VaR_fundamental[i][:])
plt.title('基础违约银行数量VaR————杠杆')
plt.xlabel('杠杆')
plt.ylabel('基础违约银行数量VaR')
plt.legend(legend)
fig3.savefig('D:/Documents/GitHub/practice/数据模型2/改变外部资产损失率：3-基础违约银行数量VaR————杠杆')

fig4 = plt.figure(dpi=300)
legend = ['外部资产损失=%f' % round(i,2) for i in loss]
for i in range(m2):
    plt.plot(leverage,Mean_fundamental[i][:])
plt.title('基础违约银行数量ES————杠杆')
plt.xlabel('杠杆')
plt.ylabel('基础违约银行数量ES')
plt.legend(legend)
fig4.savefig('D:/Documents/GitHub/practice/数据模型2/改变外部资产损失率：4-基础违约银行数量ES————杠杆')

fig5 = plt.figure(dpi=300)
legend = ['外部资产损失=%f' % round(i,2) for i in loss]
for i in range(m2):
    plt.plot(leverage,Mean_contagion[i][:])
plt.title('传染违约银行数量平均数————杠杆')
plt.xlabel('杠杆')
plt.ylabel('传染违约银行数量平均数')
plt.legend(legend)
fig5.savefig('D:/Documents/GitHub/practice/数据模型2/改变外部资产损失率：5-传染违约银行数量平均数————杠杆')

fig6 = plt.figure(dpi=300)
legend = ['外部资产损失=%f' % round(i,2) for i in loss]
for i in range(m2):
    plt.plot(leverage,Median_contagion[i][:])
plt.title('传染违约银行数量中位数————杠杆')
plt.xlabel('杠杆')
plt.ylabel('传染违约银行数量中位数')
plt.legend(legend)
fig6.savefig('D:/Documents/GitHub/practice/数据模型2/改变外部资产损失率：6-传染违约银行数量中位数————杠杆')

fig7 = plt.figure(dpi=300)
legend = ['外部资产损失=%f' % round(i,2) for i in loss]
for i in range(m2):
    plt.plot(leverage,VaR_contagion[i][:])
plt.title('传染违约银行数量VaR————杠杆')
plt.xlabel('杠杆')
plt.ylabel('传染违约银行数量VaR')
plt.legend(legend)
fig7.savefig('D:/Documents/GitHub/practice/数据模型2/改变外部资产损失率：7-传染违约银行数量VaR————杠杆')

fig8 = plt.figure(dpi=300)
legend = ['外部资产损失=%f' % round(i,2) for i in loss]
for i in range(m2):
    plt.plot(leverage,ES_contagion[i][:])
plt.title('传染违约银行数量ES————杠杆')
plt.xlabel('杠杆')
plt.ylabel('传染违约银行数量ES')
plt.legend(legend)
fig8.savefig('D:/Documents/GitHub/practice/数据模型2/改变外部资产损失率：8-传染违约银行数量ES———杠杆')

fig9 = plt.figure(dpi=300)
legend = ['外部资产损失=%f' % round(i,2) for i in loss]
for i in range(m2):
    plt.plot(leverage,Mean_default[i][:])
plt.title('总违约银行数量平均数————杠杆')
plt.xlabel('杠杆')
plt.ylabel('总违约银行数量平均数')
plt.legend(legend)
fig9.savefig('D:/Documents/GitHub/practice/数据模型2/改变外部资产损失率：9-总违约银行数量平均数————杠杆')

fig10 = plt.figure(dpi=300)
legend = ['外部资产损失=%f' % round(i,2) for i in loss]
for i in range(m2):
    plt.plot(leverage,Median_default[i][:])
plt.title('总违约银行数量中位数————杠杆')
plt.xlabel('杠杆')
plt.ylabel('总违约银行数量中位数')
plt.legend(legend)
fig10.savefig('D:/Documents/GitHub/practice/数据模型2/改变外部资产损失率：10-总违约银行数量中位数————杠杆')

fig11 = plt.figure(dpi=300)
legend = ['外部资产损失=%f' % round(i,2) for i in loss]
for i in range(m2):
    plt.plot(leverage,VaR_default[i][:])
plt.title('总违约银行数量VaR————杠杆')
plt.xlabel('杠杆')
plt.ylabel('总违约银行数量VaR')
plt.legend(legend)
fig11.savefig('D:/Documents/GitHub/practice/数据模型2/改变外部资产损失率：11-总违约银行数量VaR————杠杆')

fig12 = plt.figure(dpi=300)
legend = ['外部资产损失=%f' % round(i,2) for i in loss]
for i in range(m2):
    plt.plot(leverage,ES_default[i][:])
plt.title('总违约银行数量ES————杠杆')
plt.xlabel('杠杆')
plt.ylabel('总违约银行数量ES')
plt.legend(legend)
fig12.savefig('D:/Documents/GitHub/practice/数据模型2/改变外部资产损失率：12-总违约银行数量ES————杠杆')

fig13 = plt.figure(dpi=300)
legend = ['外部资产损失=%f' % round(i,2) for i in loss]
for i in range(m2):
    plt.plot(leverage,Mean_LGD_avg[i][:])
plt.title('平均违约损失率平均数————杠杆')
plt.xlabel('杠杆')
plt.ylabel('平均违约损失率平均数')
plt.legend(legend)
fig13.savefig('D:/Documents/GitHub/practice/数据模型2/改变外部资产损失率：13-平均违约损失率平均数————杠杆')

fig14 = plt.figure(dpi=300)
legend = ['外部资产损失=%f' % round(i,2) for i in loss]
for i in range(m2):
    plt.plot(leverage,Median_LGD_avg[i][:])
plt.title('平均违约损失率中位数————杠杆')
plt.xlabel('杠杆')
plt.ylabel('平均违约损失率中位数')
plt.legend(legend)
fig14.savefig('D:/Documents/GitHub/practice/数据模型2/改变外部资产损失率：14-平均违约损失率中位数————杠杆')

fig15 = plt.figure(dpi=300)
legend = ['外部资产损失=%f' % round(i,2) for i in loss]
for i in range(m2):
    plt.plot(leverage,VaR_LGD_avg[i][:])
plt.title('平均违约损失率VaR————杠杆')
plt.xlabel('杠杆')
plt.ylabel('平均违约损失率VaR')
plt.legend(legend)
fig15.savefig('D:/Documents/GitHub/practice/数据模型2/改变外部资产损失率：15-平均违约损失率VaR————杠杆')

fig16 = plt.figure(dpi=300)
legend = ['外部资产损失=%f' % round(i,2) for i in loss]
for i in range(m2):
    plt.plot(leverage,ES_LGD_avg[i][:])
plt.title('平均违约损失率ES————杠杆')
plt.xlabel('杠杆')
plt.ylabel('平均违约损失率ES')
plt.legend(legend)
fig16.savefig('D:/Documents/GitHub/practice/数据模型2/改变外部资产损失率：16-平均违约损失率ES————杠杆')

