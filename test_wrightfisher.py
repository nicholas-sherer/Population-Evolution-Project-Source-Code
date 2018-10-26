# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 15:14:30 2018

@author: nashere2
"""

import numpy as np
import pytest

import wrightfisher as wf


def expected_delta_mean_exp_f(pop_dist, fitnesses):
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
    scale = (sampled_delta_exp_f(pop_dist, fitnesses) /
             expected_delta_mean_exp_f(pop_dist, fitnesses))
    assert(np.abs(scale-1) < .1)


pytest.main()
