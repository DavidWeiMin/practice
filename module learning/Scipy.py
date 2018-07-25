import networkx as nx                               #导入networkx包  
import matplotlib.pyplot as plt                     #导入绘图包matplotlib（需要安装，方法见第一篇笔记）  
G1 = nx.random_graphs.barabasi_albert_graph(20,10)    #生成一个BA无标度网络G
G2 = nx.random_graphs.ws_graph(20,10)
nx.draw(G1)                                          #绘制网络G
nx.draw(G2)
plt.savefig("ba.png")                               #输出方式1: 将图像存为一个png格式的图片文件
plt.savefig("er.png")
plt.show()                                          #输出方式2: 在窗口中显示这幅图像