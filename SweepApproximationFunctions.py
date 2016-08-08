# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 14:07:19 2016

@author: Nicholas Sherer
"""

from __future__ import division

import numpy as np


def testApproxIneq(M, f_b, f_a, P_mu, K):
    """
    This returns whether or not the parameters chosen fulfill the three
    inequalities that must be satisfied for the mutation followed by sweep
    approximation to the full model to work.
    """
    M_test = testMIneq(M)
    P_mu_test = testP_muIneq(M, P_mu)
    fix_time_test = testFixIneq(M, f_b, f_a, P_mu, K)
    return M_test and P_mu_test and fix_time_test


def testMIneq(M):
    return M > (1 + np.sqrt(5))/2


def testP_muIneq(M, P_mu):
    return (1 - P_mu)*M > 1


def testFixIneq(M, f_b, f_a, P_mu, K):
    x = (1 - 1/M) > (f_a*P_mu + f_b*(1-P_mu)*(M-1)/((1-P_mu)*M-1))*K*np.log(K)
    return x


def fixationProbability(N, s):
    """
    Computes the classical fixation probability for an invader against a
    wildtype assuming no further mutations and only the only two fitnesses are
    the invader and the wildtype. The elif statement is there to handle the
    problem of float-induced error when dividing small numbers by small
    numbers.
    """
    if s > 1 / N:
        return (1 - np.exp(-2*s))/(1 - np.exp(-4*N*s))
    elif s > -1 / N:
        return 1 / (2*N)
    else:
        return 0


def effectiveFitnessDifference(f_1, f_2, mu_1, mu_2):
    return f_2 - f_1 + (mu_1 - mu_2)


def powerOfMultiple(mu1, mu2, mu_multiple):
    """
    Given mu1, mu2, and mu_multiple, this computes the k such that mu2 = mu2 *
    mu_multiple^k
    """
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
    s = effectiveFitnessDifference(0, delta_f, mu_1, mu_2)
    p_fix = fixationProbability(K, s)
    k = powerOfMultiple(mu_1, mu_2, M)
    N_eff = findN00overN(P_mu, M, mu_1, delta_f, 10, 10) * findN0k(P_mu, M, k)
    # N_eff = findN0k(P_mu, M, k)
    temp = f_b * (1 - P_mu) * mu_2 * N_eff * p_fix * K
    return temp


def findAntiMutatorSweepRate(mu_1, mu_2, delta_f, M, f_a, P_mu, K):
    s = effectiveFitnessDifference(0, 0, mu_1, mu_2)
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
