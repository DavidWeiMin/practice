import networkx as nx
import math
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
# # Networkx Basics
# G1 = nx.Graph()
# G2 = nx.DiGraph()
# G3 = nx.MultiGraph()
# G4 = nx.MultiDiGraph()
# # Graph Creation
# G1.add_edge(1, 2)                        # default edge data=1
# G1.add_edge(2, 3, weight=0.9)            # specify edge data
# G1.add_nodes_from([2, 3])
# H = nx.path_graph(10)
# G1 = nx.Graph(day="Friday")
# G1.add_node(1, time='5pm')
# G1.add_edge(1,2)
# # G1.add_edges_from([(2,3),(1,8)])
# G1.add_edges_from(H.edges)

# nx.draw(G1,with_labels=True)
# plt.show()
# # Drawing
# G = nx.cubical_graph()
# plt.subplot(121)
# nx.draw(G)                              # default spring_layout
# plt.subplot(122)
# nx.draw(G, pos=nx.circular_layout(G), nodecolor='r', edge_color='b')
# plt.show()
# # Data Structure
# G = nx.Graph()
# G.add_edge('A', 'B',color='blue',weight=0.84,size=300)
# G.add_edge('B', 'C',color='yellow')
# nx.draw(G)
# plt.show()
# print(G.adj)
# print(G.edges['A','B']['color'])
# print(G['A']['B']['size'])
# Random Graphs
#########################################################
# 规则网络
# G0 = nx.random_regular_graph(4,10)
# nx.draw(G0)
# nx.draw_circular(G0)
# nx.draw_kamada_kawai(G0)
# nx.draw_spring(G0)
# nx.draw_spectral(G0)
# plt.show()
# 随机网络
# G1 = nx.erdos_renyi_graph(10,0.7)
# # nx.draw(G1)
# nx.draw_shell(G1)
# plt.show()
# 小世界网络
G2 = nx.newman_watts_strogatz_graph(10,4,1)
nx.draw_shell(G2)
plt.show()
# BA无标度网络
G3 = nx.barabasi_albert_graph(200,1)
nx.draw(G3,node_size=50,width=0.5)
plt.show()
# A = nx.adjacency_matrix(G3)
# print(A.todense())
# A = nx.to_numpy_matrix(G3)
# print(A)


# 将networkx格式数据转化为numpy矩阵
# a = np.reshape(np.random.random_integers(0, 1, size=100), (10, 10))
# D = nx.DiGraph(a)
# nx.draw_shell(D)
# plt.show()
# a = np.ones(10)
# b = np.zeros(10)
# for i in range(9):
#     a = np.row_stack((a,b))
# print(a)
# G = nx.from_numpy_matrix(a)
# nx.draw(G)
# plt.show()


