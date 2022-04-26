import networkx as nx
import random



def make_random_graph(n, m):
    def generate_graph(n, m):
        array = []
        for i in range(n):
            array.append(i)
        graph = {}
        for i in array:
            graph[i] = sorted(random.sample(range(0, n), m))
        return graph

    g = nx.Graph(generate_graph(n, m))
    for (u, v) in g.edges():
        g.edges[u, v]['weight'] = random.randint(0, 10)

    g.remove_edges_from(nx.selfloop_edges(g))
    return g

