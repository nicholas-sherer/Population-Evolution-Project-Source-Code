# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 14:07:19 2016

@author: Nicholas Sherer
"""

from __future__ import division

import numpy as np


def fixationProbability(N, s):
    if s > 1 / N:
        return (1 - np.exp(-2*s))/(1 - np.exp(-4*N*s))
    elif s > -1 / N:
        return 1 / (2*N)
    else:
        return 0


def effectiveFitnessDifference(f_1, f_2, mu_1, mu_2):
    return f_2 - f_1 + (mu_1 - mu_2)


def powerOfMultiple(mu1, mu2, mu_multiple):
    return np.round(np.log(mu2/mu1)/np.log(mu_multiple))


def findN0k(P_mu, M, k):
    if k == 0:
        return 1
    else:
        return P_mu * M**(k-1)/(M**k - 1) * findN0k(P_mu, M, k-1)


def findNl0(P_mu, mu_min, delta_f, l):
    if l == 0:
        return 1
    else:
        return (1 - P_mu) * mu_min / (l * delta_f) *\
            findNl0(P_mu, mu_min, delta_f, l-1)


def findNlk(P_mu, M, mu_min, delta_f, k, l):
    if k == 0:
        return findNl0(P_mu, mu_min, delta_f, l)
    elif l == 0:
        return findN0k(P_mu, M, k)
    else:
        num_term1 = (1 - P_mu)*M**k*mu_min*findNlk(P_mu, M, mu_min, delta_f, k,
                                                   l-1)
        num_term2 = P_mu*M**(k-1)*mu_min*findNlk(P_mu, M, mu_min, delta_f, k-1,
                                                 l)
        denom = l*delta_f + (M**k - 1)*mu_min
        return (num_term1 + num_term2) / denom


def findN00overN(P_mu, M, mu_min, delta_f, kmax, lmax):
    total = 0
    for k in range(kmax):
        for l in range(lmax):
            total = total + findNlk(P_mu, M, mu_min, delta_f, k, l)
    return 1 / total


def findTransitionRate(mu_1, mu_2, delta_f, M, f_b, f_a, P_mu, K):
    if mu_1 <= mu_2:
        return findMutatorSweepRate(mu_1, mu_2, delta_f, M, f_b, P_mu, K)
    elif mu_1 > mu_2:
        return findAntiMutatorSweepRate(mu_1, mu_2, delta_f, M, f_a, P_mu, K)
    else:
        print('uh oh, something went wrong')


def findMutatorSweepRate(mu_1, mu_2, delta_f, M, f_b, P_mu, K):
    s = effectiveFitnessDifference(0, delta_f, mu_1, mu_2, P_mu)
    p_fix = fixationProbability(K, s)
    k = powerOfMultiple(mu_1, mu_2, M)
    N_eff = findN00overN(P_mu, M, mu_1, delta_f, 10, 10) * findN0k(P_mu, M, k)
    # N_eff = findN0k(P_mu, M, k)
    temp = f_b * (1 - P_mu) * mu_2 * N_eff * p_fix * K
    return temp


def findAntiMutatorSweepRate(mu_1, mu_2, delta_f, M, f_a, P_mu, K):
    s = effectiveFitnessDifference(0, 0, mu_1, mu_2, P_mu)
    p_fix = fixationProbability(K, s)
    k = powerOfMultiple(mu_1, mu_2, M)
    if k < -1.00:
        return 0
    else:
        N_eff = findN00overN(P_mu, M, mu_1, delta_f, 10, 10)
        # N_eff = 1
        temp = f_a * P_mu * mu_1 * N_eff * p_fix * K
        return temp


def findTransitionMatrix(mu_list, delta_f, M, f_b, f_a, P_mu, K):
    matrix_size = np.size(mu_list)
    temp = np.zeros((matrix_size, matrix_size))
    for i in range(matrix_size):
        for j in range(matrix_size):
            if i != j:
                Tij = findTransitionRate(mu_list[j], mu_list[i], delta_f, M,
                                         f_b, f_a, P_mu, K)
                temp[i, j] += Tij
                temp[j, j] -= Tij
    return temp


def findSteadyState(transition_matrix):
    t_norm = transition_matrix / np.max(transition_matrix)
    U, s, V = np.linalg.svd(t_norm)
    ss_eig = s[-1]  # numpy's svd sorts by descending order of singular values
    null_vector = V[-1, :]
    proportions = null_vector / np.sum(null_vector)
    return ss_eig, proportions
