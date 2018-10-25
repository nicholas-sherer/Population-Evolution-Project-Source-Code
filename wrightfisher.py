# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 12:05:22 2018

@author: nashere2
"""

import numpy as np
import arraymultinomial as am


def mean_exp_fitness(pop_dist, fitnesses):
    return np.sum(pop_dist*np.exp(fitnesses))/np.sum(pop_dist)


def wright_fisher_probabilities(pop_dist, fitnesses):
    return np.exp(fitnesses)/mean_exp_fitness(pop_dist, fitnesses) * \
        pop_dist/np.sum(pop_dist)


def _wright_fisher_probabilities(pop_dist, fitnesses):
    x = pop_dist*np.exp(fitnesses)
    return x/np.sum(x)


# due to a lack of error handling in np.random.multinomial this will silently
# return very wrong results if N exceeds about 10^9
def wright_fisher_fitness_update(pop_dist, fitnesses):
    probabilities = _wright_fisher_probabilities(pop_dist, fitnesses)
    shape = probabilities.shape
    probabilities.shape = probabilities.size
    next_generation = np.random.multinomial(np.sum(pop_dist), probabilities)
    next_generation.shape = shape
    return next_generation


def wright_fisher_fitness_update_bigN(pop_dist, fitnesses, checks=True):
    probabilities = _wright_fisher_probabilities(pop_dist, fitnesses)
    shape = probabilities.shape
    probabilities.shape = probabilities.size
    next_generation = am.multinomial_int64(np.sum(pop_dist),
                                           probabilities, checks)
    next_generation.shape = shape
    return next_generation
