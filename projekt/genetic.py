import matplotlib.pyplot as plt

from main import make_random_graph_with_pomeranian_voivodeship_and_Gliwice_city_names as m_r_g_w_p_v_a_G_c_n
import pygad
import networkx as nx



graph = m_r_g_w_p_v_a_G_c_n()
print(graph)
print(nx.to_dict_of_lists(graph))
print(nx.get_edge_attributes(graph, 'weight'))

S = []

#definiujemy parametry chromosomu
gene_space = {'low': 0, 'high': 8, 'step': 1}

#definiujemy funkcję fitness
def fitness_func(solution, solution_idx):
    fitness = 0
    visited = []
    position = 0
    for i in solution:
        i = int(i)
        if i in nx.to_dict_of_lists(graph)[position]:
            if (i in visited):
                fitness -= 10
            else:
                visited.append(i)
                fitness -= nx.get_edge_attributes(graph, 'weight')[min(position, i), max(position, i)]
                position = i
        else:
            fitness -= 10
    if(len(visited) is not len(graph.nodes)):
        fitness -= len(graph.nodes)*10
    return fitness

fitness_function = fitness_func

#ile chromsomów w populacji
#ile genow ma chromosom
sol_per_pop = 10
num_genes = len(graph.nodes)*2

#ile wylaniamy rodzicow do "rozmanazania" (okolo 50% populacji)
#ile pokolen
#ilu rodzicow zachowac (kilka procent)
num_parents_mating = 5
num_generations = 50
keep_parents = 2

#jaki typ selekcji rodzicow?
#sss = steady, rws=roulette, rank = rankingowa, tournament = turniejowa
parent_selection_type = "sss"

#w il =u punktach robic krzyzowanie?
crossover_type = "single_point"

#mutacja ma dzialac na ilu procent genow?
#trzeba pamietac ile genow ma chromosom
mutation_type = "random"
mutation_percent_genes = 25

#inicjacja algorytmu z powyzszymi parametrami wpisanymi w atrybuty
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

#tutaj dodatkowo wyswietlamy sume wskazana przez jedynki
print("Generations: " + str(generations))

#wyswietlenie wykresu: jak zmieniala sie ocena na przestrzeni pokolen
ga_instance.plot_fitness()