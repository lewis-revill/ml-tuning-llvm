#!/usr/bin/env python3

from benchmarks.benchmark import evaluate_benchmarks

from multiprocessing import Pool

import logging as log
import numpy as np


rng = np.random.default_rng()

default_values = np.array([0, 0, 1 << 31, 1 << 30, 0, 0, 1 << 29, 1, 1 << 24, 1])


def crossover_population(population, num_children):

  def crossover_parents(parents):
    num_values = parents[0].shape[0]
    crossover_indices = rng.choice(num_values, size=int(num_values / 2),
                                   replace=False)
    child = parents[0].copy()
    child[crossover_indices] = parents[1][crossover_indices]

    return child

  children = np.array([crossover_parents(rng.choice(population, size=2, replace=False)) for _ in range(num_children)])\

  return children


def mutate_population(population, mutation_probability):

  # Mutations occur at random for each value within an individual. A mutation
  # involves modifying that value by a random amount.
  def mutate_value(value):
    if rng.random() > mutation_probability:
      return value
    adjustment = rng.integers(low=-(1<<31), high=(1<<31)+1)
    new_value = value + adjustment
    new_value = min(new_value, (1 << 32) - 1)
    new_value = max(new_value, 0)

    return new_value

  # Apply the mutation over all values for each individual in the population.
  return np.array([[mutate_value(v) for v in i] for i in population])


def initialize_population(num_individuals):

  # Starting with the default values, perform one round of mutations to generate
  # an initial population.
  population = mutate_population(np.array([default_values for _ in range(num_individuals)]), mutation_probability=0.2)

  return population


def evaluate_fitness(individual):

  additional_llcflags = [
      '-split-stage-weight=' + str(individual[0]),
      '-memory-stage-weight=' + str(individual[1]),
      '-assign-stage-weight=' + str(individual[2]),
      '-preference-weight=' + str(individual[3]),
      '-no-preference-weight=' + str(individual[4]),
      '-local-weight=' + str(individual[5]),
      '-global-weight=' + str(individual[6]),
      '-size-weight=' + str(individual[7]),
      '-rc-priority-weight=' + str(individual[8]),
      '-mem-ops-weight=' + str(individual[9]),
  ]

  total_size = evaluate_benchmarks(additional_llcflags)

  # Lower total size is better, so invert and scale this value to obtain a
  # suitable fitness of the individual.
  return 10000 / total_size


def evaluate_population(population):

  # Evaluate the fitness of each individual in parallel.
  with Pool() as p:
    fitness_values = p.map(evaluate_fitness, population)

  return np.array(fitness_values)


def evolve(population, fitness_values, num_elite, num_to_select, mutation_probability=0.05):
  sorted_indices = fitness_values.argsort()
  sorted_population = population[sorted_indices]
  sorted_fitness_values = fitness_values[sorted_indices]

  # Create probabilities from the fitness values by normalizing using the
  # softmax function to sum to 1.
  def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)
  selection_probabilities = softmax(sorted_fitness_values)

  # Select the initial members of the new population, preferring the higher
  # fitness individuals from the current population.
  new_population = rng.choice(sorted_population, size=num_to_select,
                              p=selection_probabilities)

  # Guarantee selection of a few elite performers from the population.
  elite = sorted_population[-num_elite:]
  new_population = np.append(new_population, elite, axis=0)

  # Create new child individuals in the population by crossover of selected
  # individuals.
  num_children = population.shape[0] - new_population.shape[0]
  children = crossover_population(new_population, num_children)
  new_population = np.append(new_population, children, axis=0)

  # Mutate the individuals in the new population.
  new_population = mutate_population(new_population, mutation_probability)

  return new_population


NUM_INDIVIDUALS = 64
NUM_ELITE = 8
NUM_TO_SELECT = 16
NUM_EPOCHS = 100

def main():
  # Setup logging.
  log.basicConfig(level=log.INFO)

  # Store the average fitness and max fitness over all epochs.
  avg_fitness_history = []
  max_fitness_history = []

  # Create the initial population of individuals and evaluate their fitness.
  population = initialize_population(NUM_INDIVIDUALS)
  fitness_values = evaluate_population(population)

  # Calculate average fitness and maximum fitness.
  avg_fitness = np.sum(fitness_values, axis=0) / NUM_INDIVIDUALS
  best_index = fitness_values.argmax()
  max_fitness = fitness_values[best_index]
  best_individual = population[best_index]

  avg_fitness_history.extend([avg_fitness])
  max_fitness_history.extend([max_fitness])

  log.info('Average fitness: {avg_fitness}'.format(avg_fitness=avg_fitness))
  log.info('Max fitness: {max_fitness}'.format(max_fitness=max_fitness))
  log.info('Best individual:\n{best_individual}'.format(best_individual=best_individual))

  for i in range(NUM_EPOCHS):
    # Run through an evolution of the population, then evaluate the fitness of
    # the individuals.
    population = evolve(population, fitness_values, NUM_ELITE, NUM_TO_SELECT)
    fitness_values = evaluate_population(population)

    # Calculate average fitness and maximum fitness.
    avg_fitness = np.sum(fitness_values, axis=0) / NUM_INDIVIDUALS
    best_index = fitness_values.argmax()
    max_fitness = fitness_values[best_index]
    best_individual = population[best_index]

    avg_fitness_history.extend([avg_fitness])
    max_fitness_history.extend([max_fitness])

    log.info('Average fitness after epoch {n}: {avg_fitness}'.format(n=i+1, avg_fitness=avg_fitness))
    log.info('Max fitness after epoch {n}: {max_fitness}'.format(n=i+1, max_fitness=max_fitness))
    log.info('Best individual after epoch {n}:\n{best_individual}'.format(n=i+1, best_individual=best_individual))


if __name__ == '__main__':
  main()