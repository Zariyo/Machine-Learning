import matplotlib.pyplot as plt

from main import make_random_graph
import pygad
import networkx as nx
import time


def fitness_func(solution, solution_idx):
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



df = {}
dt = {}
# df["sss"] = []
# df["rank"] = []
# df["random"] = []
# df["tournament"] = []
# df["sus"] = []
# df["rws"] = []
# df["parents"] = []
# df["sols"] = []
# dt["sss"] = []
# dt["n"] = []
# dt["sols"] = []
# df["gens"] = []
# dt["gens"] = []

for i in range(100, 251, 10):
    df["sols"].append(i)
    dt["sols"].append(i)
    avg_sss, avg_rank, avg_random, avg_tournament, avg_sus, avg_rws = 0, 0, 0, 0, 0, 0
    avg_score = 0
    print("Computing " + str(i) + " nodes")
    fitness_function = fitness_func
    sol_per_pop = 50
    num_parents_mating = sol_per_pop//5
    num_generations = i
    keep_parents = num_parents_mating//3
    parent_selection_type = "sss"
    crossover_type = "single_point"
    mutation_type = "random"
    mutation_percent_genes = 10
    gene_space = {'low': 0, 'high': 11, 'step': 1}


    # ga_instance_rank = pygad.GA(gene_space=gene_space,
    #                             num_generations=num_generations,
    #                             num_parents_mating=num_parents_mating,
    #                             fitness_func=fitness_function,
    #                             sol_per_pop=sol_per_pop,
    #                             num_genes=num_genes,
    #                             parent_selection_type="rank",
    #                             keep_parents=keep_parents,
    #                             crossover_type=crossover_type,
    #                             mutation_type=mutation_type,
    #                             mutation_percent_genes=mutation_percent_genes)
    #
    # ga_instance_random = pygad.GA(gene_space=gene_space,
    #                               num_generations=num_generations,
    #                               num_parents_mating=num_parents_mating,
    #                               fitness_func=fitness_function,
    #                               sol_per_pop=sol_per_pop,
    #                               num_genes=num_genes,
    #                               parent_selection_type="random",
    #                               keep_parents=keep_parents,
    #                               crossover_type=crossover_type,
    #                               mutation_type=mutation_type,
    #                               mutation_percent_genes=mutation_percent_genes)
    #
    # ga_instance_tournament = pygad.GA(gene_space=gene_space,
    #                                   num_generations=num_generations,
    #                                   num_parents_mating=num_parents_mating,
    #                                   fitness_func=fitness_function,
    #                                   sol_per_pop=sol_per_pop,
    #                                   num_genes=num_genes,
    #                                   parent_selection_type="tournament",
    #                                   keep_parents=keep_parents,
    #                                   crossover_type=crossover_type,
    #                                   mutation_type=mutation_type,
    #                                   mutation_percent_genes=mutation_percent_genes)
    #
    # ga_instance_sus = pygad.GA(gene_space=gene_space,
    #                            num_generations=num_generations,
    #                            num_parents_mating=num_parents_mating,
    #                            fitness_func=fitness_function,
    #                            sol_per_pop=sol_per_pop,
    #                            num_genes=num_genes,
    #                            parent_selection_type="sus",
    #                            keep_parents=keep_parents,
    #                            crossover_type=crossover_type,
    #                            mutation_type=mutation_type,
    #                            mutation_percent_genes=mutation_percent_genes)
    #
    # ga_instance_rws = pygad.GA(gene_space=gene_space,
    #                            num_generations=num_generations,
    #                            num_parents_mating=num_parents_mating,
    #                            fitness_func=fitness_function,
    #                            sol_per_pop=sol_per_pop,
    #                            num_genes=num_genes,
    #                            parent_selection_type="rws",
    #                            keep_parents=keep_parents,
    #                            crossover_type=crossover_type,
    #                            mutation_type=mutation_type,
    #                            mutation_percent_genes=mutation_percent_genes)
    for j in range(0, 8):
        print(str(j+1) + " / 3")
        graph = make_random_graph(12, int(12 ** (1 / 2)))
        # sss
        num_genes = len(graph.nodes) * 2
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
        avg_sss += end-start
        # # rank
        # start = time.time()
        # ga_instance_rank.run()
        # end = time.time()
        # avg_rank += end-start
        # # random
        # start = time.time()
        # ga_instance_random.run()
        # end = time.time()
        # avg_random += end - start
        # # tournament
        # start = time.time()
        # ga_instance_tournament.run()
        # end = time.time()
        # avg_tournament += end - start
        # # sus
        # start = time.time()
        # ga_instance_sus.run()
        # end = time.time()
        # avg_sus += end - start
        # # rws
        # start = time.time()
        # ga_instance_rws.run()
        # end = time.time()
        avg_rws += end - start
        avg_score += -1*ga_instance.best_solution()[1]
    dt["sss"].append(avg_sss/8)
    df["sss"].append(avg_score/8)
    # df["rank"].append(avg_rank/5)
    # df["random"].append(avg_random / 5)
    # df["tournament"].append(avg_tournament / 5)
    # df["sus"].append(avg_sus / 5)
    # df["rws"].append(avg_rws / 5)

    print(df)
    print(dt)

fig, ax = plt.subplots()

ax.plot(df["parents"], df["sss"], color='red', label="sss")
# ax.plot(df["n"], df["rank"], color='yellow', label="rank")
# ax.plot(df["n"], df["random"], color='blue', label="random")
# ax.plot(df["n"], df["tournament"], color='green', label="tournament")
# ax.plot(df["n"], df["sus"], color='purple', label="sus")
# ax.plot(df["n"], df["rws"], color='orange', label="rws")
ax.set(xlabel='Amount of nodes', ylabel='# of parents',
       title='Genetic algorithm results on amount of parents')

ax.grid()
ax.legend()
plt.show()
