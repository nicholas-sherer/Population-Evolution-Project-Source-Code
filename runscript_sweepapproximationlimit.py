# -*- coding: utf-8 -*-
"""
Created on Mon May  2 00:10:38 2016

@author: Nicholas Sherer

a small script for running and storing some summary data
"""

import numpy as np

import h5py

import populationevolution_v3 as popev

init_fit_list = np.atleast_2d(np.array([0]))
init_mu_list = np.atleast_2d(np.array([.01]))
K = 10**5
init_pop_dist = np.atleast_2d(np.array([K]))

M_list = [3, 5]
P_mu_list = [.1]
delta_f_list = [.01, .04]
f_a_list = [10**-6, 10**-7]
f_b_list = [10**-6, 10**-7]

filename = 'testing_sweep_approximation.hdf5'
testfile = h5py.File(filename)

for M in M_list:
    for P_mu in P_mu_list:
        for delta_f in delta_f_list:
            for f_a in f_a_list:
                for f_b in f_b_list:
                    mu_params = [0, 0, 0, 0, 0]
                    mu_params[0] = delta_f
                    mu_params[1] = M
                    mu_params[2] = f_b
                    mu_params[3] = f_a
                    mu_params[4] = P_mu
                    testpop = popev.Population(init_fit_list, init_mu_list,
                                               init_pop_dist, mu_params, K)
                    testpopstore = popev.Population_Store(testpop, testfile,
                                                          testfile, 0)
                    testpopstore.summarystatSimStorage(0, 10**8)
