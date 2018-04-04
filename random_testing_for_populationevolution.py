# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 14:10:11 2017

@author: nashere2
"""

import numpy as np
import h5py
import populationevolution_v4 as popev


def randomPopulationParameters(N_high, pop_shape):
    f_high = np.random.uniform(0, 1.0)
    mu_low = 10**np.random.uniform(-4, -1)
    pop_dist = np.random.randint(0, N_high, pop_shape)
    K = N_high/2*pop_shape[0]*pop_shape[1]
    delta_f = np.random.uniform(.001, .1)
    mu_multiple = np.random.uniform(1.1, 5.0)
    fraction_beneficial = 10**np.random.uniform(-10, -4)
    fraction_accurate = 10**np.random.uniform(-10, -4)
    fraction_mu2mu = np.random.uniform(.001, .5)
    return f_high, mu_low, pop_dist, delta_f, mu_multiple, \
        fraction_beneficial, fraction_accurate, fraction_mu2mu, K


def testSaveandLoad(test_pop):
    with h5py.File('save_and_load_test_file.hdf5', 'w') as temp_save_file:
        test_pop_store = popev.PopulationStore(test_pop, temp_save_file,
                                               percent_memory_write=.5)
        test_pop_store.fullandsummarySimStorage(0, 100000)
    with h5py.File('save_and_load_test_file.hdf5', 'r') as temp_save_file:
        test_pop_load = \
            popev.PopulationStore.loadStartFromFile(temp_save_file,
                                                    temp_save_file['times 0 to 100000'],
                                                    temp_save_file)
        print(test_pop_load.population == test_pop)
        return test_pop, test_pop_load.population
