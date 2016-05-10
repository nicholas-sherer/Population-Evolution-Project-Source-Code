# -*- coding: utf-8 -*-
"""
Created on Tue May 10 10:34:23 2016

@author: Nicholas Sherer
"""

import numpy as np

import h5py

import populationevolution_v3 as popev

from datetime import datetime

import multiprocessing

from functools import partial

init_fit_list = np.array([0])
init_mu_list = np.array([.01])
K = 10**5
init_pop_dist = np.array([K])
mu_params = [.04, 3, 10**-4, 10**-4, .1]

testpopstores = []

runfunc = partial(popev.Population_Store.fullandsummarySimStorage,
                  t_start=0, t_finish=3000)

for i in range(5):
    filename = 'testing_code' + str(i) + repr(datetime.utcnow()) + '.hdf5'
    testfile = h5py.File(filename)
    testpop = popev.Population(init_fit_list, init_mu_list,
                               init_pop_dist, mu_params, K)
    testpopstore = popev.Population_Store(testpop, testfile, testfile, 0, .5)
    testpopstores.append(testpopstore)
    print(i)
if __name__ == '__main__':
    print(str(datetime.utcnow()))
    ped = multiprocessing.Pool(5)
    print(ped.map(runfunc, testpopstores))
    print(str(datetime.utcnow()))
