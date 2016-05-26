# -*- coding: utf-8 -*-
"""
Created on Wed May 25 22:48:26 2016

@author: Nicholas Sherer
"""

import SweepApproximationFunctions as SAF

import populationevolution_v3 as popev

import h5py

import numpy as np

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
    P_suc_est = successes / count
    P_suc_std = np.sqrt(P_suc_est*(1 - P_suc_est)/count)
    if mu_inv_ind == -1:
        s = SAF.effectiveFitnessDifference(0, 0, mu_min, mu_min / M)
    elif mu_inv_ind >= 0:
        s = SAF.effectiveFitnessDifference(0, delta_f, mu_min,
                                           mu_min*M**mu_inv_ind)
    theor_pfix = SAF.fixationProbability(K, s)
    error = np.abs(P_suc_est - theor_pfix)
    deviations = error / P_suc_std
    return (P_suc_est, P_suc_std, theor_pfix, error, deviations)
