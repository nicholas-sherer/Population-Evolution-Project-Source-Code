# -*- coding: utf-8 -*-
"""
Created on Wed May 25 22:48:26 2016

@author: Nicholas Sherer
"""

import SweepApproximationFunctions as SAF

import populationevolution_v3 as popev

import h5py

import numpy as np

import gc

# mu_min = delta_f / (2*(M-1))


def createEqSelectDist(P_mu, M, mu_min, delta_f, K):
    lmax = 10
    kmax = 10
    Nlk = np.zeros((lmax, kmax))
    for i in range(lmax):
        for j in range(kmax):
            Nlk[j, i] = SAF.findNlk(P_mu, M, mu_min, delta_f, i, j)
    Nlk = np.round(Nlk / np.sum(Nlk) * K)
    Nlk = Nlk.astype(int)
    f_list = np.linspace(0, -delta_f*lmax, num=lmax, endpoint=False)
    mu_list = mu_min * np.logspace(0, kmax, num=kmax, base=M,
                                   endpoint=False)
    return (f_list, mu_list, Nlk)


def createEqSelectDistwInv(P_mu, M, mu_min, delta_f, K, mu_inv_ind):
    (f_list, mu_list, Nlk) = createEqSelectDist(P_mu, M, mu_min, delta_f, K)
    kmax = np.size(f_list)
    lmax = np.size(mu_list)
    if mu_inv_ind < 0:
        Nlk_temp = np.zeros((lmax, kmax + 1))
        mu_list = np.insert(mu_list, 0, mu_list[0]*M**mu_inv_ind)
        Nlk_temp[0, 0] = 1
        Nlk_temp[:, 1:kmax+1] = Nlk
    elif mu_inv_ind >= 0:
        Nlk_temp = np.zeros((lmax + 1, kmax))
        f_list = np.insert(f_list, 0, f_list[0] + delta_f)
        Nlk_temp[0, mu_inv_ind] = 1
        Nlk_temp[1:lmax+1, :] = Nlk
    Nlk_temp = Nlk_temp.astype(int)
    return (f_list, mu_list, Nlk_temp)


def stopWhenInvFin(population, mu_inv, f_inv, threshold):
    if population.getNlk(f_inv, mu_inv) == 0:
        return 'Failure'
    elif population.getNlk(f_inv, mu_inv) > threshold:
        return 'Success'
    else:
        return False


def invasion(population, mu_inv_ind, threshold):
    if mu_inv_ind == -1:
        mu_inv = population.mutation_list[0]
    elif mu_inv_ind >= 0:
        mu_inv = population.mutation_list[mu_inv_ind]
    f_inv = population.fitness_list[0]
    time = 0
    while stopWhenInvFin(population, mu_inv, f_inv, threshold) is False:
        population.update()
        time = time + 1
    if stopWhenInvFin(population, mu_inv, f_inv, threshold) == 'Failure':
        return 0
    elif stopWhenInvFin(population, mu_inv, f_inv, threshold) == 'Success':
        return 1


def repeatInvasions(P_mu, M, mu_min, delta_f, K, mu_inv_ind, repeat_num):
    dists = createEqSelectDistwInv(P_mu, M, mu_min, delta_f, K, mu_inv_ind)
    f_list = np.transpose(np.atleast_2d(dists[0]))
    mu_list = dists[1]
    pop_dist = dists[2]
    if mu_inv_ind >= 0:
        threshold = pop_dist[1, 0]/2
    elif mu_inv_ind == -1:
        threshold = pop_dist[0, 1]/2
    mu_params = [delta_f, M, 0, 0, P_mu]
    count = 0
    successes = 0
    for i in range(repeat_num):
        count += 1
        pop = popev.Population(f_list, mu_list, pop_dist, mu_params, K)
        successes += invasion(pop, mu_inv_ind, threshold)
    if mu_inv_ind == -1:
        s = SAF.effectiveFitnessDifference(0, 0, mu_min, mu_min / M)
    elif mu_inv_ind >= 0:
        s = SAF.effectiveFitnessDifference(0, delta_f, mu_min,
                                           mu_min*M**mu_inv_ind)
    theor_pfix = SAF.fixationProbability(K, s)
    theor_mean = count*theor_pfix
    theor_std = np.sqrt(count*theor_pfix*(1-theor_pfix))
    error = np.abs(successes - theor_mean)
    deviations = error / theor_std
    return (theor_mean, successes, count, error, deviations)


class fixationProbTableStorage(object):

    def __init__(self, hdf5_file, params_list, repeat_num=1000):
        assert(isinstance(hdf5_file, h5py.File))
        self.hdf5_file = hdf5_file
        self.params_list = params_list
        self.theor_mean_list = []
        self.exp_mean_list = []
        self.count_list = []
        self.error_list = []
        self.deviations_from_mean_list = []
        self.repeat_num = repeat_num

    def computeFixationProbabilities(self):
        for params in self.params_list:
            P_mu = params[0]
            M = params[1]
            mu_min = params[2]
            delta_f = params[3]
            K = params[4]
            mu_inv = params[5]
            N = self.repeat_num
            answer = repeatInvasions(P_mu, M, mu_min, delta_f, K, mu_inv, N)
            self.theor_mean_list.append(answer[0])
            self.exp_mean_list.append(answer[1])
            self.count_list.append(answer[2])
            self.error_list.append(answer[3])
            self.deviations_from_mean_list.append(answer[4])

    def storeResults(self):
        np_params_arr = np.transpose(np.array(self.params_list))
        np_results_arr = np.array([self.theor_mean_list, self.exp_mean_list,
                                   self.count_list, self.error_list,
                                   self.deviations_from_mean_list])
        np_whole_arr = np.vstack((np_params_arr, np_results_arr))
        data = self.hdf5_file.create_dataset('Fixation Probabilities',
                                             shape=(11,), data=np_whole_arr,
                                             compression="gzip",
                                             compression_opts=4, shuffle=True)
        data.attrs['row1'] = 'P_mu'
        data.attrs['row2'] = 'M'
        data.attrs['row3'] = 'mu_min'
        data.attrs['row4'] = 'delta_f'
        data.attrs['row5'] = 'K'
        data.attrs['row6'] = 'mu_inv_ind'
        data.attrs['row7'] = 'theoretical_mean'
        data.attrs['row8'] = 'experimental_mean'
        data.attrs['row9'] = 'total_invasion_attempts'
        data.attrs['row10'] = 'error'
        data.attrs['row11'] = 'error_over_theoretical_standard_deviation'
        self.hdf5_file.flush()
        gc.collect()
