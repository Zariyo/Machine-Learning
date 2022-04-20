import matplotlib.pyplot as plt

from main import make_random_graph_with_pomeranian_voivodeship_and_Gliwice_city_names as m_r_g_w_p_v_a_G_c_n
import pygad
import networkx as nx
import time

graph = m_r_g_w_p_v_a_G_c_n(8, 3)
print(graph)
print(nx.to_dict_of_lists(graph))
print(nx.get_edge_attributes(graph, 'weight'))

S = []

# definiujemy parametry chromosomu
gene_space = {'low': 0, 'high': 8, 'step': 1}


# definiujemy funkcję fitness
def fitness_func(solution, solution_idx):
    fitness = 0
    visited = []
    position = 0
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


fitness_function = fitness_func
sol_per_pop = 30
num_genes = len(graph.nodes) * 2
num_parents_mating = 15
num_generations = 200
keep_parents = 5
parent_selection_type = "sss"
crossover_type = "single_point"
mutation_type = "random"
mutation_percent_genes = 25

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

ga_instance.run()

solution, solution_fitness, solution_idx = ga_instance.best_solution()
generations = ga_instance.generations_completed
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

print("Generations: " + str(generations))

# ga_instance.plot_fitness()

ga_instance_rank = pygad.GA(gene_space=gene_space,
                            num_generations=num_generations,
                            num_parents_mating=num_parents_mating,
                            fitness_func=fitness_function,
                            sol_per_pop=sol_per_pop,
                            num_genes=num_genes,
                            parent_selection_type="rank",
                            keep_parents=keep_parents,
                            crossover_type=crossover_type,
                            mutation_type=mutation_type,
                            mutation_percent_genes=mutation_percent_genes)

ga_instance_rank.run()

solution, solution_fitness, solution_idx_rank = ga_instance_rank.best_solution()
generations = ga_instance_rank.generations_completed
print("Parameters of the best solution with rank : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

print("Generations: " + str(generations))

# ga_instance.plot_fitness()
df = {}
df["sss"] = []
df["n"] = [0,10,20,30]
for i in range(0, 40, 10):
    avg = 0
    print("Computing " + str(i) + " nodes")
    gene_space = {'low': 0, 'high': i, 'step': 1}
    for j in range(0, 2):
        print(str(j+1) + " / 2")
        graph = m_r_g_w_p_v_a_G_c_n(i, int(i ** (1 / 2)))
        start = time.time()
        ga_instance.run()
        end = time.time()
        avg += end-start
    df["sss"].append(avg)

fig, ax = plt.subplots()

ax.plot(df["n"], df["sss"], color='red', label="sss")
# ax.plot(df["n"], df["rank"], color='yellow', label="rank")
# ax.plot(df["n"], df["random"], color='blue', label="random")
# ax.plot(df["n"], df["tournament"], color='green', label="tournament")
# ax.plot(df["n"], df["sss"], color='purple', label="sus")
# ax.plot(df["n"], df["rank"], color='orange', label="rws")
ax.set(xlabel='Amount of nodes', ylabel='Time (s)',
       title='Genetic algorithm time depending on crossing type')

ax.grid()
ax.legend()
plt.show()
