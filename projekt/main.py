import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random



def make_random_graph_with_pomeranian_voivodeship_and_Gliwice_city_names():
    miasta = {0: "Gdańsk", 1: "Gdynia", 2: "Sopot", 3: "Wejherowo", 4: "Reda", 5: "Lębork", 6: "Bytów", 7: "Gliwice",
              8: "Warszawa"}

    graphCities = {('Gdańsk', 'Sopot'): 6, ('Gdańsk', 'Lębork'): 8, ('Gdańsk', 'Gliwice'): 10,
                   ('Gdańsk', 'Wejherowo'): 3,
                   ('Gdańsk', 'Reda'): 5, ('Gdynia', 'Lębork'): 2, ('Gdynia', 'Bytów'): 5, ('Sopot', 'Lębork'): 1,
                   ('Sopot', 'Bytów'): 2, ('Sopot', 'Gliwice'): 1, ('Wejherowo', 'Reda'): 3, ('Wejherowo', 'Lębork'): 7,
                   ('Wejherowo', 'Bytów'): 6, ('Reda', 'Bytów'): 6, ('Reda', 'Gliwice'): 7, ('Bytów', 'Gliwice'): 8}

    def generate_graph(n, m):
        array = []
        for i in range(n):
            array.append(i)
        graph = {}
        for i in array:
            graph[i] = sorted(random.sample(range(0, n), m))
        return graph

    g = nx.Graph(generate_graph(8, 3))
    for (u, v) in g.edges():
        g.edges[u, v]['weight'] = random.randint(0, 10)

    g.remove_edges_from(nx.selfloop_edges(g))

    #g = nx.relabel_nodes(g, miasta)

    pos = nx.spring_layout(g)
    nx.draw_networkx(g, pos, with_labels=True)
    labels = nx.get_edge_attributes(g, 'weight')
    nx.draw_networkx_edge_labels(g, pos, edge_labels=labels)
    return g

