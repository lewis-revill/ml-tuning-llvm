#!/usr/bin/env python3

from benchmarks.benchmark import evaluate_benchmarks

import math
import numpy as np
import logging as log


rng = np.random.default_rng()

default_values = np.array([0, 0, 1<<31, 1<<30, 0, 0, 1<<29, 1, 1<<24, 1])


def evaluate_fitness(solution):

  additional_llcflags = [
      '-split-stage-weight=' + str(solution[0]),
      '-memory-stage-weight=' + str(solution[1]),
      '-assign-stage-weight=' + str(solution[2]),
      '-preference-weight=' + str(solution[3]),
      '-no-preference-weight=' + str(solution[4]),
      '-local-weight=' + str(solution[5]),
      '-global-weight=' + str(solution[6]),
      '-size-weight=' + str(solution[7]),
      '-rc-priority-weight=' + str(solution[8]),
      '-mem-ops-weight=' + str(solution[9]),
  ]

  total_size = evaluate_benchmarks(additional_llcflags)

  # Lower total size is better, so invert and scale this value to obtain a
  # suitable fitness of the solution.
  return 1000000 / total_size


def get_random_neighbour(solution):

  # Adjustments of each value in the solution are used to generate a
  # neighbouring solution. The adjustments are random.
  def adjust_value(value):
    adjustment = int(rng.triangular(left=-128, mode=0, right=128))
    new_value = value + adjustment
    new_value = min(new_value, (1 << 32) - 1)
    new_value = max(new_value, 0)

    return new_value

  # Apply the random adjustment over all values in the given solution.
  return np.array([adjust_value(v) for v in solution])


MIN_TEMPERATURE = 0.5

def main():
  # Setup logging.
  log.basicConfig(level=log.INFO)

  # Start with the default values and attempt to improve from that point.
  current_solution = default_values
  current_fitness = evaluate_fitness(current_solution)

  current_temperature = 1.0

  log.info('Current temperature {current_temperature}'
                .format(current_temperature=current_temperature))
  log.info('Fitness of current solution {current_fitness}'
                .format(current_fitness=current_fitness))

  best_solution = current_solution.copy()
  best_fitness = current_fitness

  while True:
    # If the temperature hits the minimum temperature then the annealing process
    # is over.
    if current_temperature <= MIN_TEMPERATURE:
      break

    # Get a neighbouring solution and evaluate its fitness.
    neighbour = get_random_neighbour(current_solution)
    neighbour_fitness = evaluate_fitness(neighbour)

    # Evaluate acceptance criteria for setting this neighbour as the current
    # solution.
    if neighbour_fitness >= current_fitness:
      current_solution = neighbour.copy()
      current_fitness = neighbour_fitness
    else:
      # Generate a probability for acceptance of a worse solution, which tends
      # towards 1 for a solution with the same fitness, and theoretically tends
      # towards zero for infinitely large differences in fitness.
      acceptance_probability = math.exp(
          100 * (neighbour_fitness - current_fitness)
          / (1 + current_temperature))

      # Decide based on this probability whether to explore this solution.
      if rng.random() < acceptance_probability:
        current_solution = neighbour.copy()
        current_fitness = neighbour_fitness

    # Update the all-time best solution if the current solution is better.
    if current_fitness > best_fitness:
      best_solution = current_solution.copy()
      best_fitness = current_fitness

    # Decrease the temperature of the system.
    current_temperature *= 0.99

    log.info('Current temperature {current_temperature}'
                 .format(current_temperature=current_temperature))
    log.info('Fitness of current solution {current_fitness}'
                 .format(current_fitness=current_fitness))
    log.info('Fitness of best solution {best_fitness}'
                 .format(best_fitness=best_fitness))


if __name__ == '__main__':
  main()
