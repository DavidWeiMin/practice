# 程序还可完善的地方
# 1、资产负债表的生成方式
# 2、传染过程可以用清算支付向量
# 3、偿还比例矩阵
import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt


# TA = 1000 * np.ones(N,1) # 银行总资产向量
# eta = 0.5 * np.ones(N,1) # 银行储户资产占银行总资产比重向量
# M = eta * TA # 银行储户资产向量
# phi = 0.05 * np.ones(N,1) # 流动性资产占总资产比重
# V = phi * TA # 流动性资产向量

def createNetwork(alpha,beta,tao,N,gamma,k):
    # 《金融市场中传染风险建模与分析》P65构建银行间网络框架
    C = np.zeros((N,N))
    c = np.zeros((N,1))
    # 用均匀分布构造幂律
    for i in range(N):
        c[i] = ((1 - random.uniform(0,1)) * (gamma - 1) / k) ** (1 - gamma)
    # c = [random.paretovariate(gamma) for i in range(N)] # 不知道怎么生成幂律
    for i in range(N):
        for j in range(N):
            C[i][j] = (c[i] ** alpha) * (c[j] ** beta)
    c_max = max(c)
    # print(c_max)
    c_min = min(c)
    # print(c_min)
    G = C >= c_max * c_min * tao
    G = G.astype(int)
    for i in range(N):
        G[i,i] = 0
    return G

def plotNetwork(array):
    G = nx.from_numpy_matrix(array)
    nx.draw(G,with_labels=True,width=0.5,node_size=200)
    plt.show()

def balanceSheet1(p,q,m,IL,G):
    # IL为银行的银行间总贷款，并且在债务银行之间均匀分布
    # 默认参数 p = 0.3, q = 0.1, m = 0.05
    TA = sum(IL) / p
    IB = np.zeros(((len(IL),1)))
    I = M = BA = NW = EA = IB
    L = np.zeros((len(IL),len(IL)))
    for i in range(len(IL)):
        if sum(G[i,:]) != 0:
            L[i,:] = IL / np.reshape(sum(G[i,:]),(1)) # 此处注意将求和后的shape进行reshape
        else:
            L[i,:] = np.zeros((1,len(IL)))
            IL[i] = 0
    # 计算资产负债表
    for i in range(len(IL)):
        IB[i] = sum(L[:,i])
        I[i] = IB[i] - IL[i] + (1 - p - m) * TA / len(IL)
        M[i] = m * (I[i] + IL[i]) /(1 - m)
        BA[i] = I[i] + M[i] + IL[i]
        NW[i] = BA[i] * q 
    EA = I + M
    return TA,L,IL,I,M,BA,EA,NW

# def balanceSheet2():
#     # 或许可以通过迭代调用函数来求得资产负债矩阵

def unique_index(s,element):
    return [index for index,value in enumerate(s) if value==element]

def addshock(shockWho):
    temp = []
    newShockWho = [0,0]
    for who in shockWho[:,1]:
        if who not in temp:
            location = unique_index(shockWho[:,1],who)
            temp = temp + [who]
            newShockWho = np.row_stack((newShockWho,[sum(shockWho[location,0]),who]))
    newShockWho = np.delete(newShockWho,0,0)
    return newShockWho

def contagion(shock,NW,L,bankrupt,G): # 可以考虑把shock和who整合成两列数据
    # shockwho都是一个矩阵，第一列是shock，第二列是who
    # 计算违约之前应当将shockwho中对同一银行的冲击进行叠加
    hasDefault = False
    shockWho = addshock(shockWho)
    shockWho[:,1] = np.round(shockWho[:,1],decimals=0)
    newShockWho = [0,0,0]
    newBankrupt = []
    for value in shockWho:
        if value[2] != '!':
            if value[0] > L[value[1],value[2]]:
                value[0] = L[value[1],value[2]]
        if value[0] > NW[int(value[1])]:
            hasDefault = True
            NW[int(value[1])] = 0
            newBankrupt.append((int(value[1])))
            # 找到每个与shockwho[i,1]有联系的银行以及遭受的冲击赋值给newShockWho
            who = unique_index(G[:,int(value[1])],1)
            if len(who) > 0:
                if value[0] - NW[int(value[1])] <= sum(L[:,int(value[1])]):
                    totalShock = value[0] - NW[int(value[1])]
                else:
                    totalShock = sum(L[:,int(value[1])])
                shock = totalShock * L[who,int(value[1])] / sum(L[who,int(value[1])])
                temp = np.column_stack((shock,who))
                newShockWho = np.row_stack((newShockWho,temp))
        else:
            NW[int(value[1])] = NW[int(value[1])] - value[0]
    # 将newBankrupt填充至1xN的向量
    if hasDefault:
        G[newBankrupt,:] = 0
        L[newBankrupt,:] = 0
        for i in range(len(newBankrupt),N):
            newBankrupt.append((-1))
        bankrupt = np.row_stack((bankrupt,newBankrupt))
        newShockWho = np.delete(newShockWho,0,0)
        newBankrupt = np.array(newBankrupt)
        if len(newShockWho) > 0:
            bankrupt = contagion(newShockWho,NW,L,bankrupt,G)
    return bankrupt


# 网络结构参数
global N
N = 5
# 资产负债表参数
IL = 1000 * np.ones(N)
G = createNetwork(0.5,1.2,0.3,N,1.35,100)
TA,L,IL,I,M,BA,EA,NW = balanceSheet1(0.3,0.1,0.05,IL,G)
shockWho = np.array([[111,1，'!'],[233,2,'!'],[111,1,'!'],[111,1,'!']])
bankrupt = contagion(shock,NW,L,np.zeros(N),G)
np.delete(bankrupt,0,0)
print(bankrupt)
# plotNetwork(G)
