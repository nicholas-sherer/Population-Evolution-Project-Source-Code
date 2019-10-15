# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 12:05:22 2018

@author: nashere2
"""

import numpy as np
from . import arraymultinomial as am


def mean_exp_fitness(pop_dist, fitnesses):
    return np.sum(pop_dist*np.exp(fitnesses))/np.sum(pop_dist)


def var_exp_fitness(pop_dist, fitnesses):
    mean_exp2_fitness = np.sum(pop_dist*np.exp(2*fitnesses))/np.sum(pop_dist)
    return mean_exp2_fitness - mean_exp_fitness(pop_dist, fitnesses)**2


def wright_fisher_probabilities(pop_dist, fitnesses):
    # subtracting the minimum fitness reduces overflow errors while
    # leaving the answer unchanged
    x = pop_dist*np.exp(fitnesses-np.min(fitnesses))
    return x/np.sum(x)


# due to a lack of error handling in np.random.multinomial this will silently
# return very wrong results if N exceeds about 10^9
def wright_fisher_fitness_update(pop_dist, fitnesses):
    probabilities = wright_fisher_probabilities(pop_dist, fitnesses)
    shape = probabilities.shape
    probabilities.shape = probabilities.size
    next_generation = np.random.multinomial(np.sum(pop_dist), probabilities)
    next_generation.shape = shape
    return next_generation


def wright_fisher_fitness_update_bigN(pop_dist, fitnesses, checks=True):
    probabilities = wright_fisher_probabilities(pop_dist, fitnesses)
    shape = probabilities.shape
    probabilities.shape = probabilities.size
    next_generation = am.multinomial_int64(np.sum(pop_dist),
                                           probabilities, checks)
    next_generation.shape = shape
    return next_generation
