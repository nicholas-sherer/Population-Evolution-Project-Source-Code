# -*- coding: utf-8 -*-
"""
Created on Thu May  5 15:51:46 2016

@author: Nicholas Sherer
"""

import numpy as np

import h5py

import populationevolution_v3 as popev

from datetime import datetime

init_fit_list = np.array([0])
init_mu_list = np.array([.01])
K = 10**5
init_pop_dist = np.array([K])
mu_params = [.04, 3, 10**-4, 10**-4, .1]

filename = 'testing_code' + repr(datetime.utcnow()) + '.hdf5'
testfile = h5py.File(filename)

testpop = popev.Population(init_fit_list, init_mu_list,
                           init_pop_dist, mu_params, K)
testpopstore = popev.Population_Store(testpop, testfile, testfile, 0, .5)
testpopstore.fullandsummarySimStorage(0, 3000)