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

from datetime import datetime
import gc
import h5py
import numpy as np
import psutil
import raggedtorectangle as r2r


class Population(object):

    """This class represents a population described by a frequency distribution
    of individuals with two traits, fitness and a mutation rate. Fitnesses are
    spread out linearly with an increment delta_fitness. Mutation rates are
    spread out logarithmically with a multiplier of mu_multiple. Mutations may
    either increase or decrease fitness or increase or decrease the mutation
    rate.
    """

    def __init__(self, f_max, mu_min, population_distribution,
                 delta_fitness, mu_multiple, fraction_beneficial,
                 fraction_accurate, fraction_mu2mu, K):
        """
        The population is described both by its state and the parameters of the
        model. Population_distribution should be a single number or a numpy
        array.
        """

        self.population_distribution = \
            np.atleast_2d(np.array(population_distribution, dtype='int64'))
        if np.any(self.population_distribution) < 0:
            raise ValueError('the population distribution must be nonnegative')
        pop_shape = self.population_distribution.shape

        self.delta_fitness = delta_fitness
        if self.delta_fitness <= 0:
            raise ValueError('delta fitness must be positive')

        self.mu_multiple = mu_multiple
        if self.mu_multiple <= 1:
            raise ValueError('mu multiple must be greater than one')

        self.fraction_beneficial = fraction_beneficial
        if self.fraction_beneficial >= 1 or self.fraction_beneficial < 0:
            raise ValueError('fraction beneficial must be >= 0 and < 1')

        self.fraction_accurate = fraction_accurate
        if self.fraction_accurate >= 1 or self.fraction_accurate < 0:
            raise ValueError('fraction accurate must be >=0 and < 1')

        self.fraction_mu2mu = fraction_mu2mu
        if self.fraction_mu2mu >= 1 or self.fraction_mu2mu < 0:
            raise ValueError('fraction_mu2mu must be >=0 and < 1')

        self.pop_cap = K
        if self.pop_cap < 100:
            raise ValueError('pop_cap must be greater than or equal to 100')

        f_min = f_max - delta_fitness*(pop_shape[0]-1)
        self.fitness_list = np.transpose(np.atleast_2d(np.linspace(f_max,
                                         f_min, pop_shape[0])))

        self.mutation_list = np.geomspace(mu_min,
                                          mu_min*mu_multiple**(pop_shape[1]-1),
                                          pop_shape[1])

        self._trimUpdates()

    @classmethod
    def arrayInit(cls, fitness_list, mutation_list,
                  population_distribution, delta_fitness,
                  mu_multiple, fraction_beneficial,
                  fraction_accurate, fraction_mu2mu, K):
        f_max = fitness_list[0]
        mu_min = mutation_list[0]
        population = cls(f_max, mu_min, population_distribution, delta_fitness,
                         mu_multiple, fraction_beneficial, fraction_accurate,
                         fraction_mu2mu, K)
        if population.population_distribution.shape != \
            (population.fitness_list.size, population.mutation_list.size):
            raise ValueError('''Shapes must be compatible''')
        population.fitness_list = np.atleast_2d(np.array(fitness_list))
        population.mutation_list = np.atleast_1d(np.array(mutation_list))
        population._trimUpdates()
        return population

    @classmethod
    def loadFromFile(cls, load_group):
        """Load a Population from the end of a run saved as an hdf5 file."""
        params_list = ['delta_fitness', 'mu_multiple', 'fraction_beneficial',
                       'fraction_accurate', 'fraction_mu2mu']
        mu_params = [load_group.attrs[attr] for attr in params_list]
        K = load_group.attrs['pop_cap']
        fitness_list = load_group['fitness_history'][:, :, -1]
        mutation_list = load_group['mutation_history'][:, :, -1][0]
        population_distribution = load_group['pop_history'][:, :, -1]
        population = cls.arrayInit(fitness_list, mutation_list,
                                   population_distribution, *mu_params, K)
        return population

    def __call__(self, fitness, mutation_rate):
        """Return number of individuals with this fitness and mutation rate."""
        l = np.where(self.fitness_list == fitness)[0][0]
        k = np.where(self.mutation_list == mutation_rate)[0][0]
        if self.population_distribution[l, k].size == 0:
            return 0
        else:
            return self.population_distribution[l, k]

    def __eq__(self, other):
        if isinstance(other, Population):
            is_equal = True
            for attribute in self.__dict__:
                is_equal = (np.all(np.equal(getattr(self, attribute),
                            getattr(other, attribute)))) and is_equal
            return is_equal
        else:
            return NotImplemented

    def __repr__(self):
        mutation_list_str = repr(self.mutation_list)
        fitness_list_str = repr(self.fitness_list)
        population_distribution_str = repr(self.population_distribution)
        parameters_str = 'delta fitness: ' + str(self.delta_fitness) + \
            ', mu multiple: ' + str(self.mu_multiple) + \
            ', fraction beneficial: ' + str(self.fraction_beneficial) + \
            ', fraction accurate: ' + str(self.fraction_accurate) + \
            ', fraction mu2mu: ' + str(self.fraction_mu2mu) + \
            ', K: ' + str(self.pop_cap)
        return_str = 'mutation list:\n' + mutation_list_str + '\n' + \
            'fitness list:\n' + fitness_list_str + '\n' + \
            'population distribution:\n' + population_distribution_str + \
            '\n' + parameters_str
        return return_str

    def update(self):
        """Evolve population one generation forward.

        This is the core function of the model. It takes one time step forward
        in the population's evolution. Mathematically, this is a wright-fisher
        birth and death scheme for a replicator-mutator type model.
        """
        self._updatePopulationDistribution()
        self._updateFitnessList()
        self._updateMutationList()
        self._trimUpdates()

    def _updateFitnessList(self):
        """Add a new highest and new lowest fitness to fitness_list."""
        num_fit = np.size(self.fitness_list)
        temp_list = np.zeros((num_fit+2, 1))
        temp_list[0] = self.fitness_list[0] + self.delta_fitness
        temp_list[-1] = self.fitness_list[-1]-self.delta_fitness
        temp_list[1:num_fit+1] = self.fitness_list
        self.fitness_list = temp_list

    def _updateMutationList(self):
        """Add a new highest and new lowest mutation rate to mutation_list."""
        num_mut = np.size(self.mutation_list)
        temp_list = np.zeros(num_mut+2)
        temp_list[0] = self.mutation_list[0] / self.mu_multiple
        temp_list[-1] = self.mutation_list[-1] * self.mu_multiple
        temp_list[1:num_mut+1] = self.mutation_list
        self.mutation_list = temp_list

    def _updatePopulationDistribution(self):
        """Evolve population_distribution one generation."""
        self._updatePopulationFitness()
        self._updatePopulationMutation()
        self._updatePopulationRegulation()
        greater_part = self.population_distribution * \
            (self.population_distribution > 10**9)
        lesser_part = self.population_distribution * \
            (self.population_distribution <= 10**9)
        self.population_distribution = \
            np.random.poisson(lesser_part).astype('int64') + \
            greater_part.astype('int64')

    def _updatePopulationFitness(self):
        """Exponential growth/decay of population_distribution.

        The exponential growth/decay in the population takes the mean fitness
        as a penalty to the growth rate to fix the population size.
        """
        mean_fitness = self.meanFitness()
        delta_fitness_list = self.fitness_list - mean_fitness
        growth_vector = np.exp(delta_fitness_list)
        self.population_distribution = \
            self.population_distribution * growth_vector

    def _updatePopulationMutation(self):
        """Call all subfunctions that deal with mutations."""
        temp_nonmut = self._nonMutants()
        temp_f_up = self._fitnessUpMutants()
        temp_f_down = self._fitnessDownMutants()
        temp_mu_up = self._mutationUpMutants()
        temp_mu_down = self._mutationDownMutants()
        self.population_distribution = \
            temp_f_up + temp_f_down + temp_mu_up + temp_mu_down + temp_nonmut

    def _nonMutants(self):
        num_fit = np.size(self.fitness_list)
        num_mut = np.size(self.mutation_list)
        non_mut = np.zeros((num_fit + 2, num_mut + 2))
        non_mut[1:num_fit + 1, 1: num_mut + 1] = \
            self.population_distribution * np.exp(-self.mutation_list)
        return non_mut

    def _fitnessUpMutants(self):
        num_fit = np.size(self.fitness_list)
        num_mut = np.size(self.mutation_list)
        f_up_mut = np.zeros((num_fit + 2, num_mut + 2))
        f_up_mut[0:num_fit, 1:num_mut+1] = \
            self.population_distribution * (1 - np.exp(-self.mutation_list)) *\
            self.fraction_beneficial * (1 - self.fraction_mu2mu)
        return f_up_mut

    def _fitnessDownMutants(self):
        num_fit = np.size(self.fitness_list)
        num_mut = np.size(self.mutation_list)
        f_down_mut = np.zeros((num_fit + 2, num_mut + 2))
        f_down_mut[2:num_fit+2, 1:num_mut+1] = \
            self.population_distribution * (1 - np.exp(-self.mutation_list)) *\
            (1 - self.fraction_beneficial) * (1 - self.fraction_mu2mu)
        return f_down_mut

    def _mutationUpMutants(self):
        num_fit = np.size(self.fitness_list)
        num_mut = np.size(self.mutation_list)
        mu_up_mut = np.zeros((num_fit + 2, num_mut + 2))
        mu_up_mut[1:num_fit+1, 2:num_mut+2] = \
            self.population_distribution * (1 - np.exp(-self.mutation_list)) *\
            (1 - self.fraction_accurate) * self.fraction_mu2mu
        return mu_up_mut

    def _mutationDownMutants(self):
        num_fit = np.size(self.fitness_list)
        num_mut = np.size(self.mutation_list)
        mu_down_mut = np.zeros((num_fit + 2, num_mut + 2))
        mu_down_mut[1:num_fit+1, 0:num_mut] = \
            self.population_distribution * (1 - np.exp(-self.mutation_list)) \
            * self.fraction_accurate * self.fraction_mu2mu
        return mu_down_mut

    def _updatePopulationRegulation(self):
        """Normalize sum of population_distribution to pop_cap.

        This term controls population size by multiplying the current
        population in each bin by the carrying capacity divided by the current
        total population.
        """
        pop_size = np.sum(self.population_distribution)
        self.population_distribution = \
            self.population_distribution * self.pop_cap / pop_size

    def _trimUpdates(self):
        """Remove rows and columns of 0's at edges of population_distribution.

        This prevents the population_distribution matrix from growing forever.
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
        """Return the mean fitness of the population."""
        mean_fitness = \
            np.sum(
                np.multiply(self.population_distribution, self.fitness_list)) \
            / np.sum(self.population_distribution)
        return mean_fitness

    def meanMutationrate(self):
        """Return the mean mutation rate of the population."""
        mean_mutationrate = \
            np.sum(
                np.multiply(self.population_distribution, self.mutation_list))\
            / np.sum(self.population_distribution)
        return mean_mutationrate

    def maxFitness(self):
        """Return the highest fitness in the population."""
        return self.fitness_list[0]

    def minFitness(self):
        """Return the lowest fitness in the population."""
        return self.fitness_list[-1]

    def maxMutationrate(self):
        """Return the highest mutation rate in the population."""
        return self.mutation_list[-1]

    def minMutationrate(self):
        """Return the lowest mutation rate in the population."""
        return self.mutation_list[0]

    def mostCommontype(self):
        """Return the mode of the population distribution.

        Only return one mode- the one with highest fitness and lowest mutation
        rate.
        """
        self.population_distribution = \
            np.atleast_2d(self.population_distribution)
        i, j = np.where(
            self.population_distribution == self.population_distribution.max())
        mode_fitness = self.fitness_list[i[0]][0]
        mode_mutationrate = self.mutation_list[j[0]]
        return (mode_fitness, mode_mutationrate)

    def modeFitness(self):
        return self.mostCommontype()[0]

    def modeMutationrate(self):
        return self.mostCommontype()[1]


class PopulationStore(object):
    """
    This object handles periodically writing data to disk in an hdf5 file in
    the proper manner.
    """

    def __init__(self, population, file, group=None, time=0,
                 percent_memory_write=10):
        """
        The simplest way to initialize a new population store object. First
        make a population object, and and hdf5 file handle and the group where
        you want to store the data.
        """
        if group is None:
            group = file
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
        population = Population.loadFromFile(load_group)
        time = load_group.attrs['t_end'] + 1
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
            t_lastwrite = t_finish

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
        perms = self.diskwriteSummary
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
        """Flush hdf5 file buffer to disk, restart blobs, and free memory.

        This cleanup prevents slowdown from Python and hdf5 using up too much
        memory without freeing it.
        """
        self.file.flush()
        self.initiateSummaryBlob()
        self.initiateFullBlob()
        gc.collect()


def readHDF5PopulationHistory(read_group, time):
    pass
