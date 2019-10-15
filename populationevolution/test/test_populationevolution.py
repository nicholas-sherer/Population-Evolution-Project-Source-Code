# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 14:10:11 2017

@author: nashere2
"""

import numpy as np
import h5py
import pytest

from .. import populationevolution_v5 as popev


def random_population_parameters(N_high, pop_shape):
    f_high = np.random.uniform(0, 1.0)
    mu_low = np.minimum(10**np.random.uniform(-4, -1), .9/1.1**pop_shape[1])
    pop_dist = np.int64(10**np.random.uniform(-4, 0, pop_shape)*N_high)
    K = np.sum(pop_dist)
    delta_f = np.random.uniform(.001, .1)
    mu_multiple = np.random.uniform(1.1,
                                    np.minimum(10, mu_low**(-1/pop_shape[1])))
    fraction_beneficial = 10**np.random.uniform(-10, -4)
    fraction_accurate = 10**np.random.uniform(-10, -4)
    fraction_mu2mu = np.random.uniform(.001, .5)
    return f_high, mu_low, pop_dist, delta_f, mu_multiple, \
        fraction_beneficial, fraction_accurate, fraction_mu2mu, K


N_max = 10**8
pop_shape_max = 9

save_and_load_params = \
    [random_population_parameters(np.random.randint(101, N_max),
                                  (np.random.randint(1, pop_shape_max),
                                   np.random.randint(1, pop_shape_max)))
     for i in range(100)]
save_and_load_pops = [popev.Population(*params) for params
                      in save_and_load_params]


@pytest.mark.parametrize("test_pop", save_and_load_pops)
def test_saveandload(test_pop):
    with h5py.File('save_and_load_test_file.hdf5', 'w') as temp_save_file:
        test_pop_store = popev.PopulationStore(test_pop, temp_save_file,
                                               percent_memory_write=.5)
        test_pop_store.fullandsummarySimStorage(0, 3)
    with h5py.File('save_and_load_test_file.hdf5', 'r') as temp_save_file:
        test_pop_load = \
            popev.PopulationStore.loadStartFromFile(temp_save_file,
                                                    temp_save_file['times 0'
                                                                   ' to 2'],
                                                    temp_save_file)
        assert(test_pop_load.population == test_pop)
