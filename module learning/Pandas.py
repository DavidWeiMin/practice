# # pandas
import pandas as pd
import matplotlib.pyplot as plt
# # # Series数据结构
# # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 
# # obj1 = pd.Series([1,2,3,4])
# # print(obj1)
# obj2 = pd.Series([1,-2,3,4],index=['a','b','c','d'])
# # print(obj2)
# # print(obj2['d'])
# # print(obj2[['b','c','d']])
# # print(obj2[obj2 > 0])
# # print('b' in obj2)
# # print(1 in obj2)
# sdata = {'ohio':35000,'texas':71000,'utah':5000}
# obj3 = pd.Series(sdata)
# # print(obj3)
# obj4 = pd.Series([1,2,3,4],index=['d','a','c','b'])
# print(obj2 + obj4)
# obj3.name = 'population'
# obj3.index.name = 'state'
# print(obj3)
# ++++++++++++++++++++++++++++++++++++++++++++++++++
# # dataframe数据结构
# data = {'state':['ohio','ohio','ohio','nevada','nevada'],'year':[2000,2001,2002,2001,2002],'pop':[1.5,1.7,3.6,2.4,2.9]}
# frame = pd.DataFrame(data)
# print(frame)
# frame = pd.DataFrame(data,columns=['year','state','pop'])
# print(frame)
# frame2 = pd.DataFrame(data,columns=['year','state','pop','debt','asset','num'])
# print(frame2)
# print(frame2.year)
# print(frame2['year'])
# frame2['debt'] = 3
# print(frame2.ix[0])
# frame2['num'] = range(5)
# print(frame2)
# val = pd.Series([2,3,4,1,5],index=[2,1,0,4,3])
# frame2['asset'] = val
# print(frame2)
# frame2['Eastern'] = frame2['state'] == 'ohio'
# print(frame2)
# # 嵌套字典构建dataframe
# pop = {'nevada':{2001:2.4,2002:2.9},'ohio':{2000:1.5,2001:1.7,2002:3.6}}
# frame3 = pd.DataFrame(pop) # 外层键作为列，内层键作为索引
# frame3.index.name = 'year'
# print(frame3)
# print(frame3.T)
# # reindex 对索引或者列进行重排
# obj4 = obj4.reindex(['a','b','c','d','e'],fill_value=0) # 空值置0
# print(obj4)
# obj4 = obj4.reindex(['b','c','a','e','d','f'],method='ffill')  # 前向填充
# print(obj4)
# frame = frame.reindex(columns=['state','year','pop']) #对列重排
# print(frame)
# frame = frame.reindex(index=[1,3,4,2,0,5],method='ffill',columns=['state','year','pop']) #同时对索引与列重排
# print(frame)
# # ix索引
# print(frame.ix[2,'state'])
# # drop删除某一行(列)或几行（列）
# frame4 = frame.drop([2,3])
# frame2 = frame2.drop(columns=['state','year','Eastern'])
# print(frame4)
# print(frame2)
# # 引用与过滤
# print(frame4.ix[1:0,'state'])
# # 函数应用与映射
# f = lambda x : x.max() - x.min()
# axis0 = frame2.apply(f)
# axis1 = frame2.apply(f,axis=1)
# print(axis0)
# print(axis1)
# def f(x):
#     return pd.Series([x.max(),x.min()],index=['max','min']) # 返回多个值组成的Series
# print(frame2.apply(f)) # 返回Series
# 存取文件
data = pd.read_excel('d:/Documents/GitHub/practice/module learning/data.xlsx',index_col='日期')
fig = plt.figure()
plt.plot(data['上证综指'])
data['上证综指'].rolling(window=5,center=False).mean().plot()
data['上证综指'].rolling(window=10,center=False).mean().plot()
data['上证综指'].rolling(window=20,center=False).mean().plot()
data['上证综指'].rolling(window=30,center=False).mean().plot()
data['上证综指'].rolling(window=60,center=False).mean().plot()
plt.legend([0,5,10,20,30,60])
plt.show()