# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 11:48:38 2016

@author: Nicholas Sherer

This class represents a population described by a frequency distribution of
individuals with two traits, fitness and a mutation rate. Fitnesses are spread
out linearly with an increment delta_fitness. Mutation rates are spread out
logarithmically with a multiplier of mu_multiple. Mutations may either increase
or decrease fitness or increase or decrease the mutation rate.
"""

from __future__ import division

import numpy as np

import psutil

import raggedtorectangle as r2r

import h5py

from datetime import datetime

import gc


class Population(object):

    """This class represents a population described by a frequency distribution
    of individuals with two traits, fitness and a mutation rate. Fitnesses are
    spread out linearly with an increment delta_fitness. Mutation rates are
    spread out logarithmically with a multiplier of mu_multiple. Mutations may
    either increase or decrease fitness or increase or decrease the mutation
    rate.
    """

    def __init__(self, fitness_list, mutation_list, population_distribution,
                 mutation_params, K):
        """
        The population is described both by its state and the parameters of the
        model. fitness_list should be a vertical numpy array. mutation_list
        should be a horizontal numpy array. population_distribution should be a
        rectangular array as tall as the fitness_list and as wide as
        mutation_list. mutation_params is a 1-d array or list and K, the
        carrying capacity, is a positive integer.
        """
        self.fitness_list = fitness_list
        self.mutation_list = mutation_list
        self.population_distribution = population_distribution
        self.delta_fitness = mutation_params[0]
        self.mu_multiple = mutation_params[1]
        self.fraction_beneficial = mutation_params[2]
        self.fraction_accurate = mutation_params[3]
        self.fraction_mu2mu = mutation_params[4]
        self.pop_cap = K

    def update(self):
        """
        This is the core function of the model. It takes one time step forward
        in the population's evolution.
        """
        self.updatePopulationDistribution()
        self.updateFitnessList()
        self.updateMutationList()
        self.trimUpdates()

    def updateFitnessList(self):
        """
        This updates the list of fitnesses by adding a new highest and new
        lowest fitness.
        """
        num_fit = np.size(self.fitness_list)
        temp_list = np.zeros((num_fit+2, 1))
        temp_list[0] = self.fitness_list[0] + self.delta_fitness
        temp_list[-1] = self.fitness_list[-1]-self.delta_fitness
        temp_list[1:num_fit+1] = self.fitness_list
        self.fitness_list = temp_list

    def updateMutationList(self):
        """
        This updates the list of mutation rates by adding a new highest and new
        lowest mutation rate.
        """
        num_mut = np.size(self.mutation_list)
        temp_list = np.zeros(num_mut+2)
        temp_list[0] = self.mutation_list[0] / self.mu_multiple
        temp_list[-1] = self.mutation_list[-1] * self.mu_multiple
        temp_list[1:num_mut+1] = self.mutation_list
        self.mutation_list = temp_list

    def updatePopulationDistribution(self):
        """
        This updates the population distribution itself using a replicator-
        mutator type scheme.
        """
        self.updatePopulationFitness()
        self.updatePopulationMutation()
        self.updatePopulationRegulation()
        self.population_distribution = \
            np.random.poisson(self.population_distribution)

    def updatePopulationFitness(self):
        """
        This step is the exponential growth/decay in the population using
        the mean fitness as a penalty to fix the population size.
        """
        mean_fitness = self.meanFitness()
        delta_fitness_list = self.fitness_list - mean_fitness
        growth_vector = np.exp(delta_fitness_list)
        self.population_distribution = \
            self.population_distribution * growth_vector

    def updatePopulationMutation(self):
        """
        This function calls all the subfunctions that deal with mutations.
        """
        temp_nonmut = self.nonMutants()
        temp_f_up = self.fitnessUpMutants()
        temp_f_down = self.fitnessDownMutants()
        temp_mu_up = self.mutationUpMutants()
        temp_mu_down = self.mutationDownMutants()
        self.population_distribution = \
            temp_f_up + temp_f_down + temp_mu_up + temp_mu_down + temp_nonmut

    def nonMutants(self):
        num_fit = np.size(self.fitness_list)
        num_mut = np.size(self.mutation_list)
        non_mut = np.zeros((num_fit + 2, num_mut + 2))
        non_mut[1:num_fit + 1, 1: num_mut + 1] = \
            self.population_distribution * np.exp(-self.mutation_list)
        return non_mut

    def fitnessUpMutants(self):
        num_fit = np.size(self.fitness_list)
        num_mut = np.size(self.mutation_list)
        f_up_mut = np.zeros((num_fit + 2, num_mut + 2))
        f_up_mut[0:num_fit, 1:num_mut+1] = \
            self.population_distribution * (1 - np.exp(-self.mutation_list)) *\
            self.fraction_beneficial * (1 - self.fraction_mu2mu)
        return f_up_mut

    def fitnessDownMutants(self):
        num_fit = np.size(self.fitness_list)
        num_mut = np.size(self.mutation_list)
        f_down_mut = np.zeros((num_fit + 2, num_mut + 2))
        f_down_mut[2:num_fit+2, 1:num_mut+1] = \
            self.population_distribution * (1 - np.exp(-self.mutation_list)) *\
            (1 - self.fraction_beneficial) * (1 - self.fraction_mu2mu)
        return f_down_mut

    def mutationUpMutants(self):
        num_fit = np.size(self.fitness_list)
        num_mut = np.size(self.mutation_list)
        mu_up_mut = np.zeros((num_fit + 2, num_mut + 2))
        mu_up_mut[1:num_fit+1, 2:num_mut+2] = \
            self.population_distribution * (1 - np.exp(-self.mutation_list)) *\
            (1 - self.fraction_accurate) * self.fraction_mu2mu
        return mu_up_mut

    def mutationDownMutants(self):
        num_fit = np.size(self.fitness_list)
        num_mut = np.size(self.mutation_list)
        mu_down_mut = np.zeros((num_fit + 2, num_mut + 2))
        mu_down_mut[1:num_fit+1, 0:num_mut] = \
            self.population_distribution * (1 - np.exp(-self.mutation_list)) \
            * self.fraction_accurate * self.fraction_mu2mu
        return mu_down_mut

    def updatePopulationRegulation(self):
        """
        This term controls population size by multiplying the current
        population in each bin by the carrying capacity divided by the current
        total population.
        """
        pop_size = np.sum(self.population_distribution)
        self.population_distribution = \
            self.population_distribution * self.pop_cap / pop_size

    def trimUpdates(self):
        """
        This function removes rows and columns at the edges with no population
        preventing the population distribution matrix from growing forever.
        """
        while np.sum(self.population_distribution[0, :]) == 0:
            self.population_distribution = \
                np.delete(self.population_distribution, 0, 0)
            self.fitness_list = np.delete(self.fitness_list, 0, 0)
        while np.sum(self.population_distribution[-1, :]) == 0:
            self.population_distribution = \
                np.delete(self.population_distribution, -1, 0)
            self.fitness_list = np.delete(self.fitness_list, -1, 0)
        while np.sum(self.population_distribution[:, 0]) == 0:
            self.population_distribution = \
                np.delete(self.population_distribution, 0, 1)
            self.mutation_list = np.delete(self.mutation_list, 0, 0)
        while np.sum(self.population_distribution[:, -1]) == 0:
            self.population_distribution = \
                np.delete(self.population_distribution, -1, 1)
            self.mutation_list = np.delete(self.mutation_list, -1, 0)

    def meanFitness(self):
        """
        Returns the mean fitness of the population.
        """
        mean_fitness = \
            np.sum(
                np.multiply(self.population_distribution, self.fitness_list)) \
            / np.sum(self.population_distribution)
        return mean_fitness

    def meanMutationrate(self):
        """
        Returns the mean mutation rate of the population
        """
        mean_mutationrate = \
            np.sum(
                np.multiply(self.population_distribution, self.mutation_list))\
            / np.sum(self.population_distribution)
        return mean_mutationrate

    def maxFitness(self):
        """
        Returns the highest fitness in the population.
        """
        return self.fitness_list[0]

    def minFitness(self):
        """
        Returns the lowest fitness in the population.
        """
        return self.fitness_list[-1]

    def maxMutationrate(self):
        """
        Returns the highest mutation rate in the population.
        """
        return self.mutation_list[-1]

    def minMutationrate(self):
        """
        Returns the lowest mutation rate in the population.
        """
        return self.mutation_list[0]

    def mostCommontype(self):
        """
        Returns the mode of the population distribution. Only returns one mode,
        choosing the one with highest fitness and lowest mutation rate.
        """
        self.population_distribution = \
            np.atleast_2d(self.population_distribution)
        i, j = np.where(
            self.population_distribution == self.population_distribution.max())
        mode_fitness = self.fitness_list[i[0]]
        mode_mutationrate = self.mutation_list[j[0]]
        return (mode_fitness, mode_mutationrate)

    def modeFitness(self):
        return self.mostCommontype()[0]

    def modeMutationrate(self):
        return self.mostCommontype()[1]

    def getNlk(self, fitness, mutation_rate):
        """
        Returns the number of individuals with a given fitness and mutation
        rate.
        """
        l = np.where(self.fitness_list == fitness)
        k = np.where(self.mutation_list == mutation_rate)
        if self.population_distribution[l[0], k].size == 0:
            return 0
        else:
            return self.population_distribution[l[0], k]


class Population_Store(object):
    """
    This object handles periodically writing data to disk in an hdf5 file in
    the proper manner.
    """

    def __init__(self, population, file, group, time, percent_memory_write=10):
        """
        The simplest way to initialize a new population store object. First
        make a population object, and and hdf5 file handle and the group where
        you want to store the data.
        """
        assert(isinstance(group, h5py.Group)), "%r isn't an hdf5 group" % group
        self.population = population
        self.group = group
        self.file = file
        self.pmw = percent_memory_write
        self.blobdata = {}
        self.time = time
        self.initiateSummaryBlob()
        self.updateSummaryBlob()
        self.initiateFullBlob()
        self.updateFullBlob()

    @classmethod
    def loadStartFromFile(cls, file, load_group, write_group, pmw=10):
        """
        This is a way of starting from the endpoint of a run previously saved
        in an hdf5 file.
        """
        population = Population(0, 0, 0, [0, 0, 0, 0, 0], 0)
        attr_list = ['delta_fitness', 'mu_multiple', 'fraction_beneficial',
                     'fraction_accurate', 'fraction_mu2mu', 'pop_cap']
        for attr in attr_list:
            population.__dict__[attr] = load_group.attrs[attr]
        dataset_list = ['pop_history', 'fitness_history', 'mutation_history']
        insert_list = ['population_distribution', 'fitness_list',
                       'mutation_list']
        for i in range(len(dataset_list)):
            dslice = load_group[dataset_list[i]][:, :, -1]
            population.__dict__[insert_list[i]] = dslice
        time = load_group.attrs['t_end'] + 1
        population.clear()
        return cls(population, file, write_group, time, pmw)

    def initiateSummaryBlob(self):
        self.blobdata['summary_stats'] = {}
        summary_stat = self.blobdata['summary_stats']
        summary_stat['mean_fitness'] = (self.population.meanFitness, [])
        summary_stat['mean_mutation'] = (self.population.meanMutationrate, [])
        summary_stat['max_fitness'] = (self.population.maxFitness, [])
        summary_stat['min_fitness'] = (self.population.minFitness, [])
        summary_stat['max_mutation'] = (self.population.maxMutationrate, [])
        summary_stat['min_mutation'] = (self.population.minMutationrate, [])
        summary_stat['mode_fitness'] = (self.population.modeFitness, [])
        summary_stat['mode_mutation'] = (self.population.modeMutationrate, [])

    def updateSummaryBlob(self):
        for item in self.blobdata['summary_stats'].values():
            item[1].append(item[0]())

    def initiateFullBlob(self):
        self.blobdata['full_distribution'] = {}
        full_stat = self.blobdata['full_distribution']
        full_stat['fitness_history'] = ('fitness_list', [])
        full_stat['mutation_history'] = ('mutation_list', [])
        full_stat['pop_history'] = ('population_distribution', [])

    def updateFullBlob(self):
        for item in self.blobdata['full_distribution'].values():
            item[1].append(np.atleast_2d(getattr(self.population, item[0])))

    def simStorage(self, t_start, t_finish, temp_store_func, perm_store_func):
        t_lastwrite = t_start
        for i in range(t_start, t_finish):
            self.population.update()
            for func in temp_store_func:
                func()
            if i % 500 == 0 and psutil.Process().memory_percent(memtype='uss')\
                    > self.pmw:
                for func in perm_store_func:
                    func(t_lastwrite, i)
                self.cleanup()
                t_lastwrite = i + 1
        if t_lastwrite < t_finish:
            for func in perm_store_func:
                func(t_lastwrite, t_finish)
            self.cleanup()

    def simStorageC(self, t_start, stop_con, temp_store_func, perm_store_func):
        t_lastwrite = t_start
        i = t_start
        while stop_con() is False:
            i += 1
            self.population.update()
            for func in temp_store_func:
                func()
            if i % 500 == 0 and psutil.Process().memory_percent(memtype='uss')\
                    > self.pmw:
                for func in perm_store_func:
                    func(t_lastwrite, i)
                self.cleanup()
                t_lastwrite = i + 1
        if t_lastwrite < i + 1:
            for func in perm_store_func:
                func(t_lastwrite, i)
            self.cleanup()

    def fullSimStorage(self, t_start, t_finish):
        temps = self.updateFullBlob
        perms = self.diskwriteFull
        self.simStorage(t_start, t_finish, temps, perms)

    def summarySimStorage(self, t_start, t_finish):
        temps = self.updateSummaryBlob
        perms = self.diskwriteFull
        self.simStorage(t_start, t_finish, temps, perms)

    def fullandsummarySimStorage(self, t_start, t_finish):
        temps = [self.updateFullBlob, self.updateSummaryBlob]
        perms = [self.diskwriteFull, self.diskwriteSummary]
        self.simStorage(t_start, t_finish, temps, perms)

    def writeAttributes(self, new_data, t_i, t_iplus):
        attr_list = ['delta_fitness', 'mu_multiple', 'fraction_beneficial',
                     'fraction_accurate', 'fraction_mu2mu', 'pop_cap']
        for attr in attr_list:
            new_data.attrs[attr] = getattr(self.population, attr)
        new_data.attrs['t_start'] = t_i
        new_data.attrs['t_end'] = t_iplus
        new_data.attrs['date'] = repr(datetime.utcnow())

    def diskwriteSummary(self, t_i, t_iplus):
        segment = 'times ' + repr(t_i) + ' to ' + repr(t_iplus) +\
            ' summary stats'
        new_data = self.group.create_group(segment)
        self.writeAttributes(new_data, t_i, t_iplus)
        for k, v in self.blobdata['summary_stats'].items():
            reg_data = np.array(v[1])
            new_data.create_dataset(k, data=reg_data, compression="gzip",
                                    compression_opts=4, shuffle=True)

    def diskwriteFull(self, t_i, t_iplus):
        segment = 'times ' + repr(t_i) + ' to ' + repr(t_iplus)
        new_data = self.group.create_group(segment)
        self.writeAttributes(new_data, t_i, t_iplus)
        for k, v in self.blobdata['full_distribution'].items():
            reg_data = r2r.raggedTo3DRectangle(v[1])
            new_data.create_dataset(k, data=reg_data, compression="gzip",
                                    compression_opts=4, shuffle=True)

    def cleanup(self):
        """
        Used to force the hdf5 file buffers to flush and restart the summary
        blobs. This prevents slowdown from Python using up too much memory.
        """
        self.file.flush()
        self.initiateSummaryBlob()
        self.initiateFullBlob()
        gc.collect()
