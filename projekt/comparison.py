import matplotlib.pyplot as plt
from main import make_random_graph
import pygad
import networkx as nx
import time
import pyswarms as ps
import math
import numpy as np
from pyswarms.utils.functions import single_obj as fx
from pyswarms.utils.plotters import plot_cost_history
from main import make_random_graph
import time
import networkx.algorithms.approximation
import random


def fitness_func(solution, solution_idx=0):
    fitness = 0
    visited = [int(solution[0])]
    position = int(solution[0])
    for i in solution:
        if len(visited) is len(graph.nodes):
            return fitness
        i = int(i)
        if i in nx.to_dict_of_lists(graph)[position]:
            if i in visited:
                fitness -= nx.get_edge_attributes(graph, 'weight')[min(position, i), max(position, i)]
                position = i
            else:
                visited.append(i)
                fitness -= nx.get_edge_attributes(graph, 'weight')[min(position, i), max(position, i)]
                position = i
        else:
            fitness -= len(graph.nodes) * 2
    if len(visited) is not len(graph.nodes):
        fitness -= len(graph.nodes) * 10
    return fitness


def fitness_func_pso(solution):
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
    j = [fitness_func_pso(x[i]) for i in range(n_particles)]
    return np.array(j)


def algorithm_fitness(graph):
    sol = networkx.algorithms.approximation.traveling_salesman_problem(graph, cycle=False)
    return fitness_func(sol), sol


fitness_function = fitness_func
sol_per_pop = 50
num_parents_mating = 5
num_generations = 50
keep_parents = 3
parent_selection_type = "sss"
crossover_type = "single_point"
mutation_type = "random"
mutation_percent_genes = 12

dt = {}
dt["genetic"] = []
dt["swarm"] = []
dt["algorithm"] = []
dt["n"] = [10, 20, 30, 40, 50]

df = {}
df["genetic"] = []
df["swarm"] = []
df["algorithm"] = []
df["n"] = [10, 20, 30, 40, 50]

options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

for i in range(10, 31, 10):
    print(i)
    gene_space = {'low': 0, 'high': i, 'step': 1}
    print("Calcualting for " + str(i) + " nodes")
    for j in range(0, 1):
        print("Calculating " + str(j) + " / 5")
        graph = make_random_graph(i, int(i ** (1 / 2)))
        num_genes = len(graph.nodes()) * 2
        ga_instance = pygad.GA(gene_space=gene_space,
                               num_generations=num_generations,
                               num_parents_mating=num_parents_mating,
                               fitness_func=fitness_function,
                               sol_per_pop=sol_per_pop,
                               num_genes=num_genes,
                               parent_selection_type=parent_selection_type,
                               keep_parents=keep_parents,
                               crossover_type=crossover_type,
                               mutation_type=mutation_type,
                               mutation_percent_genes=mutation_percent_genes)
        start = time.time()
        ga_instance.run()
        end = time.time()
        dt["genetic"].append(end - start)
        df["genetic"].append(-1*int(ga_instance.best_solution()[1]))
        x_max = np.ones(len(graph.nodes) * 2) * len(graph.nodes)
        x_min = np.zeros(len(graph.nodes) * 2)
        my_bounds = (x_min, x_max)
        optimizer = ps.single.GlobalBestPSO(n_particles=100, dimensions=len(graph.nodes) * 2,
                                            options=options, bounds=my_bounds)
        start = time.time()
        res = optimizer.optimize(f, 200)
        end = time.time()
        dt["swarm"].append(end - start)
        df["swarm"].append(int(res[0]))
        start = time.time()
        res = algorithm_fitness(graph)
        end = time.time()
        dt["algorithm"].append(end - start)
        df["algorithm"].append(-1*res[0])

print(df)
print(dt)
