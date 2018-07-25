import numpy as np
import random
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']

def gini_coef(wealths):
    cum_wealths = np.cumsum(sorted(np.append(wealths, 0)))
    sum_wealths = cum_wealths[-1]
    xarray = np.array(range(0, len(cum_wealths))) / np.float(len(cum_wealths)-1)
    yarray = cum_wealths / sum_wealths
    B = np.trapz(yarray, x=xarray)
    A = 0.5 - B
    return A / (A+B)
std = []
Median = []
gini = []
N = 100
index = set(range(N))
asset = 100 * np.ones(N)
social_attribution = dict(zip(index,asset))
for i in range(1000):
    index -= set(np.where(asset <= 0)[0])
    asset[list(index)] -= 1
    for j in index:
        candidate = index - set([j])
        asset[random.choice(list(candidate))] += 1
    std.append(np.std(asset))
    Median.append(np.median(asset))
    gini.append(gini_coef(asset))

# plt.hist(asset)
plt.title('财富分配')
plt.bar(range(N),np.sort(asset))
print('中位数：',Median[-1])
Max = max(asset)
print('最大值：',Max)
ratio_ten = sum(np.sort(asset)[-round(N * 0.1):]) / N / 100
print('最富有的10%的人财富占社会财富百分比：',ratio_ten)
ratio_twenty = sum(np.sort(asset)[-round(N * 0.2):]) / N / 100
print('最富有的20%的人财富占社会财富百分比：',ratio_twenty)
fig1 = plt.figure()
plt.title('标准差')
plt.plot(std)
fig2 = plt.figure()
plt.title('中位数')
plt.plot(Median)
fig3 = plt.figure()
plt.title('基尼系数')
plt.plot(gini)
plt.show()