def create_cost(graph,start_node):
    cost = graph.copy()
    for node in graph.keys():
        cost[node] = float('inf')
    for node in graph[start_node].keys():
        cost[node] = graph[start_node][node]
    return cost

def create_parent(graph,start_node):
    parent = graph.copy()
    for node in graph.keys():
        parent[node] = None
    for node in graph[start_node].keys():
        parent[node] = start_node
    return parent

def find_lowest_cost_node(cost,processed):
    lowest_cost = float('inf')
    lowest_cost_node = None
    for node in cost.keys() :
        if node not in processed and cost[node] < lowest_cost:
            lowest_cost = cost[node]
            lowest_cost_node = node
    return lowest_cost_node

def dijkstra(graph,start_node,final_node):
    cost = create_cost(graph,start_node)
    parent = create_parent(graph,start_node)
    processed = [start_node,final_node]
    node = find_lowest_cost_node(cost,processed)
    while node is not None:
        neighbor = graph[node]
        for n in neighbor.keys():
            new_cost = cost[node] + neighbor[n]
            if new_cost < cost[n]:
                cost[n] = new_cost
                parent[n] = node
        processed.append((node))
        node = find_lowest_cost_node(cost,processed)
    path = [final_node]
    node = final_node
    while node is not start_node:
        path.append((parent[node]))
        node = parent[node]
    return path[-1::-1],cost[final_node]

if __name__=='__main__':
    graph = {}
    start_node = 'start'
    final_node = 'final'
    graph[start_node] = {'a':5,'b':2}
    graph['a'] = {'c':4,'d':2}
    graph['b'] = {'a':8,'d':7}
    graph['c'] = {'d':6,'final':3}
    graph['d'] = {'final':1}
    graph[final_node] = {}
    path,cost = dijkstra(graph,start_node,final_node)
    print('最短路径：',path)
    print('最小成本：',cost)
