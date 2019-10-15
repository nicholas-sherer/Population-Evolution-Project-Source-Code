# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 15:14:30 2018

@author: nashere2
"""

import numpy as np
import pytest

from .. import wrightfisher as wf


def expected_delta_mean_exp_f(pop_dist, fitnesses):
    '''
    The expected change in fitness after one generation of wright-fisher
    evolution is equal to the variance in exponential fitness in the population
    divided by the mean exponential fitness of the population.
    '''
    return (wf.var_exp_fitness(pop_dist, fitnesses) /
            wf.mean_exp_fitness(pop_dist, fitnesses))


def sampled_delta_exp_f(pop_dist, fitnesses):
    next_gen = wf.wright_fisher_fitness_update_bigN(pop_dist, fitnesses)
    return (wf.mean_exp_fitness(next_gen, fitnesses) -
            wf.mean_exp_fitness(pop_dist, fitnesses))


def random_pop_and_fitnessses(N_scale, N_dim, f_min, f_max):
    pop = np.int64(np.random.randint(1, 5, N_dim))*N_scale
    fitnesses = np.random.uniform(f_min, f_max, N_dim)
    return pop, fitnesses


@pytest.mark.parametrize("pop_dist,fitnesses",
                         [random_pop_and_fitnessses(10**9,
                                                    np.random.randint(5, 50),
                                                    -10,
                                                    10) for i in range(1000)])
def test_delta_mean_exp_f(pop_dist, fitnesses):
    '''
    Test that the change in mean fitness from generation to generation is close
    to the expectation for the wright fisher model.
    '''
    scale = (sampled_delta_exp_f(pop_dist, fitnesses) /
             expected_delta_mean_exp_f(pop_dist, fitnesses))
    assert(np.abs(scale-1) < .1)


@pytest.mark.parametrize("pop_dist,fitnesses,trans",
                         [random_pop_and_fitnessses(10**9,
                                                    np.random.randint(5, 50),
                                                    -10,
                                                    10) +
                          (np.random.uniform(-10**6, 10**6),)
                          for i in range(1000)])
def test_f_approximately_translationally_invariant(pop_dist, fitnesses, trans):
    wf_prob = wf.wright_fisher_probabilities(pop_dist, fitnesses)
    wf_prob_trans = wf.wright_fisher_probabilities(pop_dist, fitnesses + trans)
    assert(np.allclose(wf_prob, wf_prob_trans))


#pytest.main()
