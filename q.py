# print('I\'m OK')

# if 1 > 0:
#     print('true')
# else:
#     print('false')

# a = 1
# boolean = True
# chars = 'hello world'
# print(chars)
# PI = 3.14
# print(PI)

# s1 = PI // a
# print(s1)
# s2 = PI / a
# print(s2)
# classmates = ['a', 'b', 'd']
# print(classmates)
# l = len(classmates)
# print(l)
# c0 = classmates[0]
# print(c0)
# c3 = classmates[-1]
# print(c3)
# classmates.insert(2, 'c')
# print(classmates)
# a,b,c,d = 1,True,'a',1.1
# print(a,b,c,d)
# print(type(a),type(b),type(c),type(d))
# for i in classmates:
#     print(i)

# sum = 0
# n = 99
# while n > 0:
#     sum = sum + n
#     n = n - 2
# print(sum)
# d = {'Michael': 95, 'Bob': 75, 'Tracy': 85}
# print(d['Michael'])

# a = 10
# print(type(a))
# a = 1.1
# print(type(a))
# print(a.__dir__())
# name = 'daiweimin'
# print(name.capitalize())
# name = 'dai wei min'
# print(name.split())
# print(name.find('dai'))
# print(name.find('a'))
# print(name.find('wei'))
# print(name.find('z'))
# print(name.replace(' ','|'))
# print(name.count('i',3,7))

# import datetime as dt
# print(dt.datetime.now())
# print(dt.datetime.today().weekday())
# print(dt.datetime.now().toordinal())

# t = (1,2,'sas',1.1)
# print(t)
# print(t.count('sas'))
# print(t.index('sas'))
# L = [1,2,'sas',1.1]
# print(L[2])

# a = [1,2,3,4]
# for element in a[0:2]:
#     print(element ** 2)
# r = list(range(0, 7, 1))
# print(r)
# for element in range(7):
#     print(element ** 2)

# def f(x):
#     return x ** 2
# print(f(2))
# # map(f,range(5)生成的不是list对象
# print(list(map(f,range(5))))

# 匿名函数的用法
# print(list(map(lambda x:x ** 2,range(5))))
# def even(x):
#     return x % 2 == 0
# print(list(map(even, range(9))))

# 过滤器的用法
# print(list(filter(even, range(9))))
# d = {'name':'daiweimin','height':180.1,'age':21,'birthday':'1997-01-14',}
# # print(d['name'])
# # print(type(d))
# # print(d.keys())
# # print(d.values())
# # print(d.items())

# # 集合
# A = set([1,2,3,4])
# B = set([2,3,4,5])
# # A并B
# print(A.union(B))
# # A交B
# print(A.intersection(B))
# # A减B
# print(A.difference(B))
# # A,B的对称差集
# print(A.symmetric_difference(B))

# # 矩阵
# v = [1,2,3,4]
# m = [v,v,v]
# m[1][1]=0
# print(m)
# print(m[1])
# print(m[1][1])
# print(v)
# from copy import deepcopy
# m = 3 * [deepcopy(v),]
# m[1][1]=0
# print(m)
# print(m[1])
# print(m[1][1])
# print(v)

# # NumPy
import numpy as np
# a = np.array([1,2,3,4])
# print(type(a))
# print(a.sum())
# print(a.std())
# print(a.cumsum())
# b = a * 2
# c = a ** 2
# d = np.array([a,b,c])
# print(b)
# print(c)
# print(d)
# print(d[1][2])
# zero = np.zeros((3,5),dtype='i',order='C')
# print(zero)

# 结构数组
import numpy as np
# 定义类型
# dt = np.dtype([('name','S10'),('age','i4'),('height','f'),('children/pets','i4',3)])
# s = np.array([('dwm',21,188.8,(2,2,1)),('hll',14,166.3,(1,1,1)),('dr',12,144.6,(1,2,3))],dtype=dt)
# print(s)

# r = np.random.standard_normal((2,2))
# s = np.random.standard_normal((2,2))
# print(r)
# print(s)
import matplotlib as mpl
import matplotlib.pyplot as plt
# y = np.random.standard_normal(20)
# plt.figure(figsize=(7,4))
# plt.plot(y.cumsum(),'b',lw=1.5)
# plt.plot(y.cumsum(),'ro')
# plt.grid(True)
# plt.axis('tight')
# plt.xlabel('index')
# plt.ylabel('value')
# plt.title('a simple plot')
# plt.show()
# x =np.array([1,4,9,16,25,36])
# mpl.pyplot.plot(x)
# plt.show()

# y = np.random.standard_normal((30,2))
# y[:,0] = y[:,0] * 100
# fig, ax1 = plt.subplots()
# plt.plot(y[:,0].cumsum(),'b',lw=1.5,label='1st')
# plt.plot(y[:,0].cumsum(),'ro')
# plt.legend(loc=9)
# plt.ylabel('value 1st')
# ax2 = ax1.twinx()
# plt.plot(y[:,1].cumsum(),'g',lw=1.5,label='2nd')
# plt.plot(y[:,1].cumsum(),'ro')
# plt.legend(loc=0)
# plt.ylabel('value 2nd')
# plt.grid(True)
# plt.axis('tight')
# plt.xlabel('index')
# plt.title('a simple plot')
# plt.show()

# import matplotlib.finance as mpf
# # 散点图画法1
# y = np.random.standard_normal((30,2))
# plt.figure(figsize=(7,4))
# plt.plot(y[:,0],y[:,1],'ro')
# plt.grid(True)
# plt.axis('tight')
# plt.xlabel('index')
# plt.ylabel('value')
# plt.title('scatter plot')
# plt.show()

# # 散点图画法2
# y = np.random.standard_normal((30,2))
# plt.figure(figsize=(7,4))
# plt.scatter(y[:,0],y[:,1],marker='+')
# plt.grid(True)
# plt.axis('tight')
# plt.xlabel('index')
# plt.ylabel('value')
# plt.title('scatter plot')
# plt.show()

# # colorbar
# y = np.random.standard_normal((30,2))
# c = np.random.randint(0,10,len(y))
# plt.figure(figsize=(7,5))
# plt.scatter(y[:,0],y[:,1],c=c,marker='+')
# plt.colorbar()
# plt.grid(True)
# plt.axis('tight')
# plt.xlabel('1st')
# plt.ylabel('2nd')
# plt.title('scatter plot')
# plt.show()

# y = np.random.standard_exponential((10,2))
# plt.figure(figsize=(7,5))
# plt.hist(y,label=['1st','2nd'],color=['b','g'],bins=10)
# plt.axis('tight')
# plt.legend(loc=0)
# plt.xlabel('value')
# plt.ylabel('frequence')
# plt.title('histogram plot')
# plt.show()

# pandas实战
# ----------------------------------------------------------------------------------------------
import pandas as pd
# df = pd.DataFrame([1,2,3,4],columns=['numbers'],index=['a','b','c','d']) # 创建Dataframe（适合创建批量数据）
# df2 = pd.DataFrame({'numbers':5,'floats':5.5,'names':'gjs'},index=['e']) # 创建Dataframe（适合创建单个数据）
# print(df)
# print(df2)
# print(df.index)
# print(df.columns)
# print(df.ix['b'])
# print(df.ix[['a','c']]) # 多重指标选择(注意多重指标应当以list输入，所以要加[])
# print(df.ix[df.index[2:3]]) # 经索引对象选择
# print(df.sum()) # 按列求和
# print(df.apply(lambda x:x ** 2)) # 对每一个元素平方
# print(df ** 2) # 对每一个元素平方
# df['floats'] = (1.1,2.2,3.3,4.4) # 增加floats列，改变了df
# print(df['floats'])
# df['names'] = pd.DataFrame(['dwm','waq','zxs','qrw'],index=['d','a','c','b']) #使用DataFrame来增加列数据
# print(df)
# 附加数据(行),没改变df
# print(df.append({'numbers':5,'floats':5.5,'names':'gjs'},ignore_index=True)) # 增加的数据用{}包裹，忽略索引会使得生成的数据索引变为纯数字
# print(df) # df并没有被改变，因为新生成的DataFrame没有被赋值给df
# df = df.append(pd.DataFrame({'numbers':5,'floats':5.5,'names':'gjs'},index=['e']))
# print(df)
# 用join增加列数据（能处理遗漏数据的意外情况）
# print(df.join(pd.DataFrame([1,4,9,16,25],index=['a','b','c','d','z'],columns=['squares']))) # columns也要以list形式输入
# df3 = df.join(pd.DataFrame([1,4,9,16,25],index=['a','b','c','d','z'],columns=['squares']),how='outer') # 使用两个索引值的并集
# df4 = df.join(pd.DataFrame([1,4,9,16,25],index=['a','b','c','d','z'],columns=['squares']),how='inner') # 使用两个索引值的交集
# df5 = df.join(pd.DataFrame([1,4,9,16,25],index=['a','b','c','d','z'],columns=['squares']),how='left') # 使用调用方法的对象中的索引值
# df6 = df.join(pd.DataFrame([1,4,9,16,25],index=['a','b','c','d','z'],columns=['squares']),how='right') # 使用被连接对象的索引值
# print(df3)
# print(df4)
# print(df5)
# print(df6)
# print(df[['numbers','floats']].mean())
# print(df[['numbers','floats']].std())

# a = np.random.standard_normal((9,4))
# a = a.round(3)
# print(a)
# df = pd.DataFrame(a) # 利用ndarray创建DataFrame对象
# print(df) # 索引与列名默认为range(n)
# df.columns = [['No1','No2','No3','No4']] # 修改列名
# dates = pd.date_range('2015-1-1',periods=9,freq='M') # 创建时间序列
# df.index = dates # 修改索引为时间索引
# print(df)
# pandas Dataframe 内建方法
# print(df.sum())
# print(df.mean())
# print(df.median())
# print(df.describe()) # 显示基本统计量
# print(np.sqrt(df)) # 对DaraFrame对象使用NumPy通用函数
# print(np.sqrt(df).sum())
# print(np.sqrt(df.sum())) # 比较这两个表达式
# pandas使用matplotlib
# df.cumsum().plot(lw=2.0)
# plt.show()

# Series类
# s = df['No1'] # 得到一个Series类
# print(type(s))
# print(type(df))

# GroupBy 操作
# df['Quarter'] = ['Q1','Q1','Q1','Q2','Q2','Q2','Q3','Q3','Q3'] # 为了分组添加一列，表示对应时间索引所属的季度
# print(df)
# groups = df.groupby(['Quarter'])
# print(groups.max())
# print(groups.mean())
# print(groups.size())
# print(groups.describe())
# 多列分组
# df['Even'] = ['Odd','Even','Odd','Even','Odd','Even','Odd','Even','Odd']
# groups = df.groupby(['Quarter','Even'])
# print(groups.size())

# I/O
from random import gauss
# a = [gauss(1.5,2) for i in range(1000)]
path = 'H:/'
import pickle
# pkl_file = open(path+'data.pkl','wb+')
# pickle.dump(a, pkl_file)
# pkl_file.close()

# pkl_file = open(path+'data.pkl','r')
# a = pickle.load(pkl_file)

# 读写csv文件
# # ------------------------------------------------------------------------------------
# rows = 100
# a = np.random.standard_normal((rows, 5))
# a = a.round(3)
# csv_file = open(path + 'data2.csv', 'w')
# header = 'No1,No2,No3,No4,No5\n'
# csv_file.write(header)
# for (No1, No2, No3, No4, No5) in a: # for No1, No2, No3, No4, No5 in a:也行
#     s = '%f,%f,%f,%f,%f\n' % (No1, No2, No3, No4, No5)
#     csv_file.write(s)
# csv_file.close()

# rows = 100
# a = np.random.standard_normal((rows, 5))
# a = a.round(3)
# dates = pd.date_range('2014-1-1', periods=rows, freq='H')
# csv_file = open(path + 'data.csv', 'w')
# header = 'date,No1,No2,No3,No4,No5\n'
# csv_file.write(header)
# for dates_,(No1, No2, No3, No4, No5) in zip(dates, a):
#     s = '%s,%f,%f,%f,%f,%f\n' % (dates_, No1, No2, No3, No4, No5)
#     csv_file.write(s)
# csv_file.close()
# # ------------------------------------------------------------------------------


# df = pd.DataFrame(np.random.randn(6,4), index=list('abcdef'), columns=list('ABCD'))
# print(df[:3])
# print(df['a':'d'])
# print(df[[True,True,True,False,True,False]]) # 前三行（布尔数组长度等于行数）
# print(df[df['A']>0]) # A列值大于0的行
# print(df[(df['A']>0) | (df['B']>0)]) # A列值大于0，或者B列大于0的行
# print(df[(df['A']>0) & (df['C']>0)]) # A列值大于0，并且C列大于0的行
# print('\n')
# print(df.loc[['a','c','d'],'A'])


print('都为你解开你的夸奖我s')
input()