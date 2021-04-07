#!/usr/bin/env python3

from benchmarks.benchmark import evaluate_benchmarks

from multiprocessing import Pool

import logging as log
import numpy as np


rng = np.random.default_rng()

MAX_VALUE = 1 << 32 - 1


def initialize_particles(num_particles):
  # Create a set of particles randomly distributed throughout the problem space.
  return rng.integers(low=0, high=MAX_VALUE, size=(num_particles, 10))


def initialize_velocities(num_particles):
  # Create a set of random velocity vectors for every particle.
  return rng.integers(low=-128, high=128, size=(num_particles, 10))


def evaluate_fitness(particle):

  additional_llcflags = [
      '-split-stage-weight=' + str(particle[0]),
      '-memory-stage-weight=' + str(particle[1]),
      '-assign-stage-weight=' + str(particle[2]),
      '-preference-weight=' + str(particle[3]),
      '-no-preference-weight=' + str(particle[4]),
      '-local-weight=' + str(particle[5]),
      '-global-weight=' + str(particle[6]),
      '-size-weight=' + str(particle[7]),
      '-rc-priority-weight=' + str(particle[8]),
      '-mem-ops-weight=' + str(particle[9]),
  ]

  total_size = evaluate_benchmarks(additional_llcflags)

  # Lower total size is better, so invert and scale this value to obtain a
  # suitable fitness of the particle.
  return 1000000 / total_size


def evaluate_particles(particles):
  # Evaluate the fitness of each particle in parallel.
  with Pool() as p:
    scores = p.map(evaluate_fitness, particles)

  return np.array(scores)


def update_personal_bests(personal_bests, particles, scores):

  best_locations, best_scores = personal_bests

  candidate_indices = scores > best_scores

  best_locations[candidate_indices] = particles[candidate_indices]
  best_scores[candidate_indices] = scores[candidate_indices]

  return best_locations, best_scores


def update_global_best(global_best, personal_bests):
  best_location, best_score = global_best

  best_locations, best_scores = personal_bests

  candidate_index = np.argmax(best_scores)
  if best_scores[candidate_index] > best_score:
    best_location = best_locations[candidate_index]
    best_score = best_scores[candidate_index]

  return best_location, best_score


def update_velocities(velocities, particles, personal_bests, global_best, omega, c1, c2):

  random_coeff_a = rng.random() * 2
  random_coeff_b = rng.random() * 2

  new_velocities = omega * velocities + c1 * random_coeff_a * (personal_bests[0] - particles) + c2 * random_coeff_b * (global_best[0] - particles)

  return new_velocities

def normalize_particles(particles):
  np.clip(particles, 0, MAX_VALUE, out=particles)

  return particles.astype(int)


def update_particles(particles, velocities):
  new_particles = particles + velocities

  new_particles = normalize_particles(new_particles)

  return new_particles


def update(particles, scores, velocities, personal_bests, global_best, omega=0.75, c1=2.05, c2=2.05):

  personal_bests = update_personal_bests(personal_bests, particles, scores)
  global_best = update_global_best(global_best, personal_bests)
  velocities = update_velocities(velocities, particles, personal_bests, global_best, omega, c1, c2)
  particles = update_particles(particles, velocities)

  return particles, velocities, personal_bests, global_best


NUM_PARTICLES = 256
NUM_ITERATIONS = 100

def main():
  # Setup logging.
  log.basicConfig(level=log.INFO)

  # Store the average fitness and max fitness over all epochs.
  avg_fitness_history = []
  max_fitness_history = []

  # Create the initial set of particles and evaluate their fitness.
  particles = initialize_particles(NUM_PARTICLES)
  scores = evaluate_particles(particles)

  # Calculate average fitness and maximum fitness.
  avg_fitness = np.sum(scores, axis=0) / NUM_PARTICLES
  best_index = scores.argmax()
  max_fitness = scores[best_index]
  best_particle = particles[best_index]

  avg_fitness_history.extend([avg_fitness])
  max_fitness_history.extend([max_fitness])

  log.info('Average fitness: {avg_fitness}'.format(avg_fitness=avg_fitness))
  log.info('Max fitness: {max_fitness}'.format(max_fitness=max_fitness))
  log.info('Best particle:\n{best_particle}'.format(best_particle=best_particle))

  velocities = initialize_velocities(NUM_PARTICLES)
  personal_bests = (particles, scores)
  global_best = (best_particle, max_fitness)

  for i in range(NUM_ITERATIONS):
    omega = 0.5 * ((i - NUM_ITERATIONS) / NUM_ITERATIONS) ** 2 + 0.4
    c1 = -3 * i / NUM_ITERATIONS + 3.55
    c2 = 3 * i / NUM_ITERATIONS + 0.55

    log.info('Omega={omega}, C1={c1}, C2={c2}'.format(omega=omega, c1=c1, c2=c2))

    particles, velocities, personal_bests, global_best = update(particles, scores, velocities, personal_bests, global_best, omega, c1, c2)
    scores = evaluate_particles(particles)

    avg_fitness = np.sum(scores, axis=0) / NUM_PARTICLES
    best_index = scores.argmax()
    max_fitness = scores[best_index]
    best_particle = particles[best_index]

    avg_fitness_history.extend([avg_fitness])
    max_fitness_history.extend([max_fitness])

    log.info('Average fitness: {avg_fitness}'.format(avg_fitness=avg_fitness))
    log.info('Max fitness: {max_fitness}'.format(max_fitness=max_fitness))
    log.info('Best particle:\n{best_particle}'.format(best_particle=best_particle))




if __name__ == '__main__':
  main()