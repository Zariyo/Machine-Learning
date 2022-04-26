import networkx.algorithms.approximation
from main import make_random_graph
import networkx as nx

graph = make_random_graph(8, 3)

g = graph

pos = nx.spring_layout(g)
nx.draw_networkx(g, pos, with_labels=True)
labels = nx.get_edge_attributes(g, 'weight')
nx.draw_networkx_edge_labels(g, pos, edge_labels=labels)


def fitness_func(solution):
    fitness = 0
    visited = [solution[0]]
    position = solution[0]
    for i in solution:
        i = int(i)
        if len(visited) is len(graph.nodes):
            return fitness
        i = int(i)
        if i in nx.to_dict_of_lists(graph)[position]:
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
        print(visited)
        print(graph.nodes)
        fitness += len(graph.nodes) * 10
    return fitness



def algorithm_fitness(graph):
    sol = networkx.algorithms.approximation.traveling_salesman_problem(graph, cycle=False)
    return (fitness_func(sol), sol)


print(algorithm_fitness(graph))
