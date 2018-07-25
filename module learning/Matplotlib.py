import matplotlib.pyplot as plt
from numpy.random import randn
# 创建figure对象
fig = plt.figure()
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
ax1.plot(randn(50).cumsum(),'k.--')
ax2.hist(randn(50),alpha=0.3,bins=20,color='b')
ax3.scatter(randn(50),2 + randn(50),color='r')
plt.subplots_adjust(wspace=0.2,hspace=0.5)
plt.show()