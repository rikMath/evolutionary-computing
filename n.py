import random
import numpy as np

# Define o número de indivíduos na população
POPULATION_SIZE = 100

# Define o número máximo de gerações
MAX_GENERATIONS = 250

# Definir o número de variáveis de decisão
NUM_VARIABLES = 2

# Definir os limites das variáveis de decisão
BOUNDS = [(0, 1), (0, 1)]

# Função objetivo
def objective_function(x):
    f1 = x[0]
    f2 = 1.0 - np.sqrt(x[0])
    return [f1, f2]

# Inicialização da população
def initialize_population(size, bounds):
    population = []
    for _ in range(size):
        individual = [random.uniform(bounds[i][0], bounds[i][1]) for i in range(len(bounds))]
        population.append(individual)
    return population

# Função de cruzamento
def crossover(parent1, parent2):
    alpha = random.random()
    offspring1 = [alpha * p1 + (1 - alpha) * p2 for p1, p2 in zip(parent1, parent2)]
    offspring2 = [(1 - alpha) * p1 + alpha * p2 for p1, p2 in zip(parent1, parent2)]
    return offspring1, offspring2

# Função de mutação
def mutate(individual, bounds, mutation_rate=0.1):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = random.uniform(bounds[i][0], bounds[i][1])
    return individual

# Algoritmo de ordenação não-dominada
def non_dominated_sorting(population, objectives):
    fronts = [[]]
    domination_counts = [0] * len(population)
    dominated_sets = [[] for _ in range(len(population))]
    ranks = [0] * len(population)
    
    for p in range(len(population)):
        for q in range(len(population)):
            if dominates(objectives[p], objectives[q]):
                dominated_sets[p].append(q)
            elif dominates(objectives[q], objectives[p]):
                domination_counts[p] += 1
        if domination_counts[p] == 0:
            ranks[p] = 0
            fronts[0].append(p)
    
    i = 0
    while len(fronts[i]) > 0:
        next_front = []
        for p in fronts[i]:
            for q in dominated_sets[p]:
                domination_counts[q] -= 1
                if domination_counts[q] == 0:
                    ranks[q] = i + 1
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    
    return fronts[:-1]

# Função de comparação de dominância
def dominates(ind1, ind2):
    return all(x <= y for x, y in zip(ind1, ind2)) and any(x < y for x, y in zip(ind1, ind2))

# Calcula a distância de amontoamento
def crowding_distance(front, objectives):
    distances = [0] * len(front)
    for i in range(len(objectives[0])):
        sorted_front = sorted(front, key=lambda x: objectives[x][i])
        distances[sorted_front[0]] = distances[sorted_front[-1]] = float('inf')
        for j in range(1, len(sorted_front) - 1):
            distances[sorted_front[j]] += (objectives[sorted_front[j + 1]][i] - objectives[sorted_front[j - 1]][i])
    return distances

# Seleção por torneio
def tournament_selection(population, distances, k=2):
    selected = []
    for _ in range(len(population)):
        contenders = random.sample(range(len(population)), k)
        contenders.sort(key=lambda x: distances[x])
        selected.append(contenders[0])
    return [population[i] for i in selected]

# Função principal do NSGA-II
def nsga2():
    population = initialize_population(POPULATION_SIZE, BOUNDS)
    objectives = [objective_function(ind) for ind in population]
    
    for generation in range(MAX_GENERATIONS):
        offspring_population = []
        
        # Realizar cruzamento e mutação
        for _ in range(POPULATION_SIZE // 2):
            parent1, parent2 = random.sample(population, 2)
            offspring1, offspring2 = crossover(parent1, parent2)
            offspring1 = mutate(offspring1, BOUNDS)
            offspring2 = mutate(offspring2, BOUNDS)
            offspring_population.extend([offspring1, offspring2])
        
        combined_population = population + offspring_population
        combined_objectives = [objective_function(ind) for ind in combined_population]
        
        fronts = non_dominated_sorting(combined_population, combined_objectives)
        new_population = []
        
        for front in fronts:
            if len(new_population) + len(front) > POPULATION_SIZE:
                distances = crowding_distance(front, combined_objectives)
                sorted_front = sorted(front, key=lambda x: distances[x], reverse=True)
                new_population.extend([combined_population[i] for i in sorted_front[:POPULATION_SIZE - len(new_population)]])
                break
            new_population.extend([combined_population[i] for i in front])
        
        population = new_population
        objectives = [objective_function(ind) for ind in population]
    
    return population, objectives

# Executa o algoritmo NSGA-II
final_population, final_objectives = nsga2()

# Imprime os resultados
for ind, obj in zip(final_population, final_objectives):
    print(f"Indivíduo: {ind}, Objetivos: {obj}")

import matplotlib.pyplot as plt

# Plota os resultados

plt.scatter([obj[0] for obj in final_objectives], [obj[1] for obj in final_objectives])
plt.xlabel("F1")
plt.ylabel("F2")
plt.title("Frente de Pareto")
plt.show()

