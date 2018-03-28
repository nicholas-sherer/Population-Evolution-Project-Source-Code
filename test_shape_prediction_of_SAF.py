# -*- coding: utf-8 -*-
"""
Created on Tue May 10 12:43:18 2016

@author: Nicholas Sherer
"""

import numpy as np

import h5py

import populationevolution_v3 as popev

from datetime import datetime

import SweepApproximationFunctions as SAF


init_fit_list = np.array([0])
mu_min = .01
init_mu_list = np.array([mu_min])
K = 10**5
init_pop_dist = np.array([K])
delta_f = .2
M = 1.7
P_mu = .05
mu_params = [delta_f, M, 0, 0, P_mu]

filename = 'testing_code' + repr(datetime.utcnow()) + '.hdf5'
testfile = h5py.File(filename)

testpop = popev.Population(init_fit_list, init_mu_list,
                           init_pop_dist, mu_params, K)
testpopstore = popev.Population_Store(testpop, testfile, testfile, 0, 50)
testpopstore.fullandsummarySimStorage(0, 10000)

fullgrop = testfile['times 0 to 10000']
pop_hist = fullgrop['pop_history'][:]
pop_dist_av = np.mean(pop_hist, 2)
trunc_pop_dist = pop_dist_av[0:5, 0:5] / np.sum(pop_dist_av)

N00 = SAF.findN00overN(P_mu, M, mu_min, delta_f, 10, 10)
lmax = 5
kmax = 5
Nlk = np.zeros((lmax, kmax))

for i in range(lmax):
    for j in range(kmax):
        Nlk[j, i] = SAF.findNlk(P_mu, M, mu_min, delta_f, i, j)*N00

print(np.sum(Nlk))
Nlk = Nlk / np.sum(Nlk)

dif = Nlk - trunc_pop_dist

np.set_printoptions(precision=2)

total_error = np.sum(np.abs(dif))/2
print('The predicted distribution of N(f,mu) is:')
print(Nlk)
print('The experimental distribution of N(f, mu) is:')
print(trunc_pop_dist)
print('The difference is:')
print(dif)
print('The total fraction of the population misassigned is:')
print(total_error)
