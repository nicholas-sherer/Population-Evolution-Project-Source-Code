# -*- coding: utf-8 -*-
"""
Created on Tue May 10 10:34:23 2016

@author: Nicholas Sherer
"""

import numpy as np

import h5py

import populationevolution_v5 as popev

from datetime import datetime

import multiprocessing

import os


class multiPopDummy(object):
    """
    Since hdf5 file handles aren't pickleable this is just the stuff
    you need to make a popteststore object with file and group name strings
    instead of hdf5 objects
    """
    def __init__(self, population, filename, time, groupname=None, pmw=5):
        self.population = population
        self.filename = filename
        self.groupname = groupname
        self.time = time
        self.pmw = pmw


def dummytoReal(dummy):
    hdf5file = h5py.File(dummy.filename)
    if dummy.groupname is not None:
        if dummy.groupname not in list(hdf5file.keys()):
            hdf5file.create_group(dummy.groupname)
        hdf5group = hdf5file[dummy.groupname]
        real = popev.PopulationStore(dummy.population, hdf5file, hdf5group,
                                     dummy.time, dummy.pmw)
    else:
        real = popev.PopulationStore(dummy.population, hdf5file, hdf5file,
                                     dummy.time, dummy.pmw)
    return real


def multiprocSimFunc(dummy):
    popstore = dummytoReal(dummy)
    popstore.fullandsummarySimStorage(0, dummy.time)
    popstore.file.close()
    return 1


mu_params_list = [0.1, 2, 0, 1/257, 0.3, 200]
init_fit_list = np.array([0])
init_mu_list = np.array([.01])
K_list = 200*2**np.arange(0,8,dtype='int64')
dummy_list = []
replicates = 10

for i in range(replicates):
    for num, K in enumerate(K_list):
        mu_params_list[-1] = K
        init_mu_list = np.array([4/K])
        init_pop_dist = np.array([K])
        filename = 'drift_barrierK{0}replicate{1}{2}.hdf5'.format(K, i, repr(datetime.utcnow()))
        testpop = popev.Population(init_fit_list, init_mu_list,
                                   init_pop_dist, *mu_params_list)
        dummy_list.append(multiPopDummy(testpop, filename, 10**7))

if __name__ == '__main__':
    print(str(datetime.utcnow()))
    ped = multiprocessing.Pool(5)
    result = ped.map(multiprocSimFunc, dummy_list)
    ped.close()
    ped.join()
    print(str(datetime.utcnow()))
