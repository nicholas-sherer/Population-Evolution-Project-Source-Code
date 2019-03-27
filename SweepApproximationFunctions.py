# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 14:07:19 2016

@author: Nicholas Sherer
"""


import functools
import numpy as np


def testApproxIneq(delta_f, M, f_b, f_a, P_mu, K):
    """
    This returns whether or not the parameters chosen fulfill the three
    inequalities that must be satisfied for the mutation followed by sweep
    approximation to the full model to work.
    """
    M_test = M_inequality(M)
    P_mu_test = P_mu_inequality(M, P_mu)
    fix_time_test = fixation_ineqality(M, f_b, f_a, P_mu, K)
    drift_test = delta_f_inequality(delta_f, K)
    return M_test and P_mu_test and fix_time_test and drift_test


def M_inequality(M):
    return M > (1 + np.sqrt(5))/2


def P_mu_inequality(M, P_mu):
    return (1 - P_mu)*M > 1


def fixation_ineqality(M, f_b, f_a, P_mu, K):
    return (1 - 1/M) > \
        (f_a*P_mu + f_b*(1-P_mu)*(M-1)/((1-P_mu)*M-1))*K*np.log(K)


def delta_f_inequality(delta_f, K):
    return delta_f > 2/K


def fixation_probability(K, s):
    """
    Computes the classical fixation probability for an invader against a
    wildtype assuming no further mutations and only the only two fitnesses are
    the invader and the wildtype. The elif statement is there to handle the
    problem of float-induced error when dividing small numbers by small
    numbers.
    """
    if s > 1 / K:
        return (1 - np.exp(-2*s))/(1 - np.exp(-2*K*s))
    elif s > -1 / K:
        return 1 / K
    else:
        return 0


def effectiveFitnessDifference(f_1, f_2, mu_1, mu_2):
    return np.exp(f_2-f_1)*(1-mu_2)/(1-mu_1)-1


def powerOfMultiple(mu1, mu2, M):
    """
    Given mu1, mu2, and M, this computes the k such that mu2 = mu1 *
    M^k
    """
    return np.round(np.log(mu2/mu1)/np.log(M))


@functools.lru_cache(maxsize=16)
def findN0l(P_mu, M, l):
    if l == 0:
        return 1
    else:
        return P_mu * M**(l-1)/(M**l - 1) * findN0l(P_mu, M, l-1)


@functools.lru_cache(maxsize=16)
def findNk0(P_mu, mu_min, delta_f, k):
    if k == 0:
        return 1
    else:
        return ((1 - P_mu) * mu_min * np.exp(delta_f) /
                ((1-mu_min)*(np.exp(k*delta_f)-1)) *
                findNk0(P_mu, mu_min, delta_f, k-1))

@functools.lru_cache(maxsize=256)
def findNkl(P_mu, M, mu_min, delta_f, k, l):
    if l == 0:
        return findNk0(P_mu, mu_min, delta_f, k)
    elif k == 0:
        return findN0l(P_mu, M, l)
    else:
        num_term1 = M**(l-1)*mu_min*P_mu*findNkl(P_mu, M, mu_min, delta_f, k,
                                                   l-1)
        num_term2 = M**l*mu_min*(1-P_mu)*np.exp(delta_f)* \
            findNkl(P_mu, M, mu_min, delta_f, k-1,
                                                 l)
        denom = np.exp(k*delta_f)*(1-mu_min) + (M**l*mu_min - 1)
        return (num_term1 + num_term2) / denom


def findNeq(P_mu, M, mu_min, delta_f, K):
    Nkl = [[1, findN0l(P_mu, M, 1)],
           [findNk0(P_mu, mu_min, delta_f, 1), findNkl(P_mu, M, mu_min, delta_f, 1, 1)]]
    total = 1
    total_next = np.sum(Nkl)
    conv = (total_next - total)/total
    k=1
    l=1
    while conv > 1/K:
        total = total_next
        last_row_sum = sum(Nkl[-1])
        last_column_sum = sum(x[-1] for x in Nkl)
        if last_row_sum > last_column_sum:
            Nkl.append([findNkl(P_mu, M, mu_min, delta_f, k+1, x) for x in range(l+1)])
            k = k+1
        else:
            for i, x in enumerate(Nkl):
                x.append(findNkl(P_mu, M, mu_min, delta_f, i, l+1))
            l = l+1
        total_next = np.sum(Nkl)
        conv = (total_next - total)/total
    return np.array(Nkl)*K/total_next


def findTransitionRate(mu_1, mu_2, delta_f, M, f_b, f_a, P_mu, K):
    if mu_1 <= mu_2:
        return findMutatorSweepRate(mu_1, mu_2, delta_f, M, f_b, P_mu, K)
    elif mu_1 > mu_2:
        return findAntiMutatorSweepRate(mu_1, mu_2, delta_f, M, f_a, P_mu, K)
    else:
        print('uh oh, something went wrong')


def findMutatorSweepRate(mu_1, mu_2, delta_f, M, f_b, P_mu, K):
    s = effectiveFitnessDifference(0, delta_f, mu_1, mu_2)
    p_fix = fixation_probability(K, s)
    k = int(powerOfMultiple(mu_1, mu_2, M))
    N_eff = findNeq(P_mu, M, mu_1, delta_f, K)[0,k]
    return f_b * (1 - P_mu) * mu_2 * N_eff * p_fix


def findAntiMutatorSweepRate(mu_1, mu_2, delta_f, M, f_a, P_mu, K):
    s = effectiveFitnessDifference(0, 0, mu_1, mu_2)
    p_fix = fixation_probability(K, s)
    k = powerOfMultiple(mu_1, mu_2, M)
    if k < -1.00:
        return 0
    else:
        N_eff = findNeq(P_mu, M, mu_1, delta_f, K)[0,0]
        return f_a * P_mu * mu_1 * N_eff * p_fix


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
