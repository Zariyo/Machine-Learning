import pygad
import numpy
import time
import math

labirynt = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],
            [1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1],
            [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
            [1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1],
            [1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1],
            [1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1],
            [1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1],
            [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]




S = []

# definiujemy parametry chromosomu
gene_space = [0,1,2,3]


# definiujemy funkcję fitness
def fitness_func(solution, solution_idx):
    x=1
    y=1
    for i in solution:
        if i == 0:
            if labirynt[y-1][x]==1:
                pass
            elif labirynt[y-1][x]==0:
                y-=1
        if i == 1:
            if labirynt[y][x+1]==1:
                pass
            elif labirynt[y][x+1]==0:
                x+=1
            elif labirynt[y][x+1]==2:
                return 0
        if i == 2:
            if labirynt[y+1][x]==1:
                pass
            elif labirynt[y+1][x]==0:
                y+=1
            elif labirynt[y+1][x]==2:
                return 0
        if i == 3:
            if labirynt[y][x-1]==1:
                pass
            elif labirynt[y][x-1]==0:
                x-=1
    fitness = -((11 - x) + (11 - y))
    return fitness


fitness_function = fitness_func

# ile chromsomów w populacji
# ile genow ma chromosom
sol_per_pop = 30
num_genes = len(S)

# ile wylaniamy rodzicow do "rozmanazania" (okolo 50% populacji)
# ile pokolen
# ilu rodzicow zachowac (kilka procent)
num_parents_mating = 10
num_generations = 5000
keep_parents = 2

# jaki typ selekcji rodzicow?
# sss = steady, rws=roulette, rank = rankingowa, tournament = turniejowa
parent_selection_type = "sss"

# w il =u punktach robic krzyzowanie?
crossover_type = "single_point"

# mutacja ma dzialac na ilu procent genow?
# trzeba pamietac ile genow ma chromosom
mutation_type = "random"
mutation_percent_genes = 13

# inicjacja algorytmu z powyzszymi parametrami wpisanymi w atrybuty
ga_instance = pygad.GA(gene_space=gene_space,
                       num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop,
                       num_genes=30,
                       parent_selection_type=parent_selection_type,
                       keep_parents=keep_parents,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes,
                       stop_criteria=["reach_0"])

# uruchomienie algorytmu
ga_instance.run()

# podsumowanie: najlepsze znalezione rozwiazanie (chromosom+ocena)
solution, solution_fitness, solution_idx = ga_instance.best_solution()
generations = ga_instance.generations_completed
print("Parameters of the best solution : {solution}".format(solution=solution))
directions = []
for i in solution:
    if i==0:
        directions.append("Gora")
    if i==1:
        directions.append("Prawo")
    if i==2:
        directions.append("Dol")
    if i==3:
        directions.append("Lewo")

print(directions)
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

# tutaj dodatkowo wyswietlamy sume wskazana przez jedynki
print("Generations: " + str(generations))

# wyswietlenie wykresu: jak zmieniala sie ocena na przestrzeni pokolen
ga_instance.plot_fitness()
