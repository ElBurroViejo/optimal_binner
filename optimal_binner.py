import numpy as np
import pandas as pd
import random
import logging

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger()
ANY_MUTATION_CHANCE = 0.02
ADDITION_CHANCE = 0.5
DELETION_CHANCE = 0.5
MINIMAL_BINNING = 1
MINIMAL_BIN_CONTENT = 5
LOW_CONTENT_CORRECTION_FACTOR = 100
POPULATION_SIZE = 1000

def mutator(input_edges: list) -> list:

    output_edges = [input_edges[0]]
    for edge in input_edges[1:-1]:
        if random.random() < ANY_MUTATION_CHANCE:
            if random.random() > DELETION_CHANCE:
                output_edges.append(edge)
            if random.random() < ADDITION_CHANCE:
                output_edges.append(
                    random.randrange(
                        min(input_edges), max(input_edges), MINIMAL_BINNING
                    )
                )
        else:
            output_edges.append(edge)
    output_edges.append(input_edges[-1])
    return sorted(list(set(output_edges)))


def crossover(mother: list, father: list):
    position = random.randrange(1, min(len(mother), len(father)), 1)
    return mother[:position] + father[position:], father[:position] + mother[position:]


def calculate_variance(edge_list: list, evaluation_list: list):

    df = pd.DataFrame({"raw_data": evaluation_list})
    return sum(df.raw_data.groupby(pd.cut(df.raw_data, edge_list)).var().dropna()), df

def fitness(edge_list: list, evaluation_list: list) -> float:
    base_var, df = calculate_variance(edge_list, evaluation_list)
    low_content =  LOW_CONTENT_CORRECTION_FACTOR * (sum(df.raw_data.groupby(pd.cut(df.raw_data, edge_list)).size()< MINIMAL_BIN_CONTENT))
    return (1/ (base_var + low_content + 0.00001))  


test_data = np.concatenate(
    [
        np.random.poisson(65, 30),
        [80, 80, 80, 80, 80, 80, 80],
        np.random.poisson(100, 15),
    ]
)

def create_sample(base_edges: list) -> list:
    
    return [mutator(base_edges) for _ in range(POPULATION_SIZE)]


def evaluate_population(population: list, test_data: list) -> list:
    return [fitness(pop, test_data) for pop in population]




def evolution(population: np.array, test_data: list) -> list: 

    #evaluate population first
    fitness = evaluate_population(population, test_data)
    fitness = np.array(fitness) / sum(fitness)
    best_score =  max(fitness)
    for i in range(POPULATION_SIZE):
        if fitness[i] == best_score:
            best_var, df = calculate_variance(population[i], test_data)
            break
    logger.info(f"Best fitness is: {best_score}, best variance : {best_var}")
    new_population = []
    
    for _ in range(int(POPULATION_SIZE/2)):
        mother = np.random.choice(range(POPULATION_SIZE), p = fitness)
        father = np.random.choice(range(POPULATION_SIZE), p = fitness)

        child_1, child2 = crossover(population[mother], population[father])
        new_population.append(mutator(child_1))
        new_population.append(mutator(child2))
    return np.array(new_population)

population = np.array(create_sample(np.arange(0, 200, 50)))


for epoche in range(100):
    logger.info(f"Epoche : {epoche}")
    population = evolution(population, test_data)