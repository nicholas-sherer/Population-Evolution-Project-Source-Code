# -*- coding: utf-8 -*-
"""
Created on Tue May 10 10:34:23 2016

@author: Nicholas Sherer
"""

import numpy as np

import h5py

import populationevolution_v4 as popev

from datetime import datetime

import multiprocessing


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

init_fit_list = np.array([0])
init_mu_list = np.array([.01])
K = 10**5
init_pop_dist = np.array([K])
mu_params = [.04, 3, 10**-4, 10**-4, .1]
dummy_list = []

for i in range(4):
    filename = 'testing_code' + str(i) + repr(datetime.utcnow()) + '.hdf5'
    testpop = popev.Population(init_fit_list, init_mu_list,
                               init_pop_dist, *mu_params, K)
    dummy_list.append(multiPopDummy(testpop, filename, 30000))

if __name__ == '__main__':
    print(str(datetime.utcnow()))
    ped = multiprocessing.Pool(6)
    result = ped.map(multiprocSimFunc, dummy_list)
    ped.close()
    ped.join()
    print(str(datetime.utcnow()))
