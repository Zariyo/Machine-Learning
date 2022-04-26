import pyswarms as ps
import math
import numpy as np
from pyswarms.utils.functions import single_obj as fx
from pyswarms.utils.plotters import plot_cost_history
import matplotlib.pyplot as plt

from main import make_random_graph
import networkx as nx
import time

graph = make_random_graph(8, 3)


def fitness_func(solution):
    fitness = 0
    visited = [int(solution[0])]
    position = int(solution[0])
    for i in solution:
        i = int(i)
        if len(visited) is len(graph.nodes):
            return fitness
        if int(i) in nx.to_dict_of_lists(graph)[position]:
            if i in visited:
                fitness += nx.get_edge_attributes(graph, 'weight')[min(position, i), max(position, i)]
                position = i
            else:
                visited.append(i)
                fitness += nx.get_edge_attributes(graph, 'weight')[min(position, i), max(position, i)]
                position = i
        else:
            fitness += len(graph.nodes) * 2
    if len(visited) is not len(graph.nodes):
        fitness += len(graph.nodes) * 10
    return fitness


def f(x):
    n_particles = x.shape[0]
    j = [fitness_func(x[i]) for i in range(n_particles)]
    return np.array(j)


x_max = np.ones(len(graph.nodes) * 2) * len(graph.nodes)
x_min = np.zeros(len(graph.nodes) * 2)
my_bounds = (x_min, x_max)

options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
optimizer = ps.single.GlobalBestPSO(n_particles=20, dimensions=len(graph.nodes) * 2,
                                    options=options, bounds=my_bounds)
optimizer.optimize(f, iters=30)

cost_history = optimizer.cost_history

print(optimizer.optimize(f, 30)[0])

# Plot!
# plot_cost_history(cost_history)
# plt.show()

df = {}
df["n"] = [10, 20, 30, 40, 50]
df["pso"] = []
# for i in range(10, 51, 10):
#     print("Computing " + str(i) + " nodes")
#     options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
#     avg = 0
#     for j in range(0, 3):
#         print(str(j + 1) + " / 2")
#         graph = make_random_graph(i, int(i ** (1 / 2)))
#         x_max = np.ones(len(graph.nodes) * 2) * len(graph.nodes)
#         x_min = np.zeros(len(graph.nodes) * 2)
#         my_bounds = (x_min, x_max)
#         optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=len(graph.nodes) * 2,
#                                             options=options, bounds=my_bounds)
#         # sss
#         start = time.time()
#         optimizer.optimize(f, 20)
#         end = time.time()
#         avg += end - start
#
#     df["pso"].append(avg / 3)
#
# print(df)
#
# fig, ax = plt.subplots()
#
# ax.plot(df["n"], df["pso"], color='red', label="pso")
# ax.set(xlabel='Amount of nodes', ylabel='Time (s)',
#        title='Global pso algorithm')
#
# ax.grid()
# ax.legend()
# plt.show()
