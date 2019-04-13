# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 14:07:19 2016

@author: Nicholas Sherer
"""

from collections import defaultdict
import functools
import numpy as np


def sweep_approx_ineq(delta_f, M, f_b, f_a, P_mu, K):
    """
    This returns whether or not the parameters chosen fulfill the three
    inequalities that must be satisfied for the mutation followed by sweep
    approximation to the full model to work.
    """
    M_test = M_inequality(M)
    P_mu_test = P_mu_inequality(M, P_mu)
    fix_time_test = fixation_inequality(M, f_b, f_a, P_mu, K)
    drift_test = delta_f_inequality(delta_f, K)
    drift_barrier_test = outside_drift_barrier_inequality(delta_f, M, f_b,
                                                          f_a, P_mu, K)
    return (M_test and P_mu_test and fix_time_test and drift_test and
            drift_barrier_test)


def M_inequality(M):
    return M > (1 + np.sqrt(5))/2


def P_mu_inequality(M, P_mu):
    return (1 - P_mu)*M > 1


def fixation_inequality(M, f_b, f_a, P_mu, K):
    return (1 - 1/M) > \
        (f_a*P_mu + f_b*(1-P_mu)*(M-1)/((1-P_mu)*M-1))*K*np.log(K)


def delta_f_inequality(delta_f, K):
    return delta_f > 2/K


def outside_drift_barrier_inequality(delta_f, M, f_b, f_a, P_mu, K):
    mu_db = mu_drift_barrier(M, f_a, K)
    mu_sw = mu_sweep(delta_f, M, f_b, f_a, P_mu)
    return (mu_sw > M**2*mu_db) and (mu_sw > 10*mu_db)


def mu_drift_barrier(M, f_a, K):
    return np.log((1-f_a)/f_a)/(2*K*np.log(M))


def mu_sweep(delta_f, M, f_b, f_a, P_mu):
    return (1 - P_mu)*f_b*delta_f/((M-1)**2*f_a + (1-P_mu)*(M-1)*f_b)


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


def effective_fitness_difference(f_1, f_2, mu_1, mu_2):
    return np.exp(f_2-f_1)*(1-mu_2)/(1-mu_1)-1


def power_of_multiple(mu1, mu2, M):
    """
    Given mu1, mu2, and M, this computes the k such that mu2 = mu1 *
    M^k
    """
    return np.log(mu2/mu1)/np.log(M)


@functools.lru_cache(maxsize=16)
def findN0l(P_mu, M, mu_min, l):
    if mu_min >= 1:
        raise ValueError('mu_min must be less than 1')
    if l == 0:
        return 1
    elif M**(l-1)*mu_min >= 1:
        return 0
    elif M**l*mu_min >=1:
        return P_mu*M**(l-1)*mu_min/(1-mu_min) * findN0l(P_mu, M, mu_min, l-1)
    else:
        return P_mu * M**(l-1)/(M**l - 1) * findN0l(P_mu, M, mu_min, l-1)


@functools.lru_cache(maxsize=16)
def findNk0(P_mu, mu_min, delta_f, k):
    if mu_min >= 1:
        raise ValueError('mu_min must be less than 1')
    if k == 0:
        return 1
    else:
        return ((1 - P_mu) * mu_min * np.exp(delta_f) /
                ((1-mu_min)*(np.exp(k*delta_f)-1)) *
                findNk0(P_mu, mu_min, delta_f, k-1))

@functools.lru_cache(maxsize=256)
def findNkl(P_mu, M, mu_min, delta_f, k, l):
    if mu_min >= 1:
        raise ValueError('mu_min must be less than 1')
    if l == 0:
        return findNk0(P_mu, mu_min, delta_f, k)
    elif k == 0:
        return findN0l(P_mu, M, mu_min, l)
    elif M**(l-1)*mu_min >= 1:
        return 0
    elif M**l*mu_min >=1:
        return ((P_mu*M**(l-1)*mu_min*findNkl(P_mu, M, mu_min, delta_f, k, l-1)
                + np.exp(delta_f)*findNkl(P_mu, M, mu_min, delta_f, k-1, l))/
                (np.exp(k*delta_f)*(1-mu_min)))
    else:
        num_term1 = M**(l-1)*mu_min*P_mu*findNkl(P_mu, M, mu_min, delta_f, k,
                                                   l-1)
        num_term2 = M**l*mu_min*(1-P_mu)*np.exp(delta_f)* \
            findNkl(P_mu, M, mu_min, delta_f, k-1,
                                                 l)
        denom = np.exp(k*delta_f)*(1-mu_min) + (M**l*mu_min - 1)
        return (num_term1 + num_term2) / denom


def findNeq(mu_min, delta_f, M, P_mu, K):
    Nkl = [[1, findN0l(P_mu, M, mu_min, 1)],
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
            # only add a column if the maximum mutation rate is below 1
            if M**l*mu_min < 1:
                for i, x in enumerate(Nkl):
                    x.append(findNkl(P_mu, M, mu_min, delta_f, i, l+1))
                l = l+1
            # otherwise add a row
            else:
                Nkl.append([findNkl(P_mu, M, mu_min, delta_f, k+1, x)
                            for x in range(l+1)])
                k = k+1
        total_next = np.sum(Nkl)
        conv = (total_next - total)/total
    return np.array(Nkl)*K/total_next


def transition_rate(mu_1, mu_2, delta_f, M, f_b, f_a, P_mu, K):
    if mu_1 <= mu_2:
        return mutator_sweeprate(mu_1, mu_2, delta_f, M, f_b, P_mu, K)
    elif mu_1 > mu_2:
        return antimutator_sweeprate(mu_1, mu_2, delta_f, M, f_a, P_mu, K)
    else:
        print('uh oh, something went wrong')


def mutator_sweeprate(mu_1, mu_2, delta_f, M, f_b, P_mu, K):
    s = effective_fitness_difference(0, delta_f, mu_1, mu_2)
    p_fix = fixation_probability(K, s)
    k = int(np.round(power_of_multiple(mu_1, mu_2, M)))
    try:
        N_eff = findNeq(mu_1, delta_f, M, P_mu, K)[0,k]
    except IndexError:
        N_eff = 0
    return f_b * (1 - P_mu) * mu_2 * N_eff * p_fix


def antimutator_sweeprate(mu_1, mu_2, delta_f, M, f_a, P_mu, K):
    s = effective_fitness_difference(0, 0, mu_1, mu_2)
    p_fix = fixation_probability(K, s)
    k = power_of_multiple(mu_1, mu_2, M)
    # in this approximation, antimutators aren't more than one multiple below
    if np.round(k) < -1.00:
        return 0
    else:
        try:
            N_eff = findNeq(mu_1, delta_f, M, P_mu, K)[0,0]
        except IndexError:
            N_eff = 0
        return f_a * P_mu * mu_1 * N_eff * p_fix


def mu_sweep_max(mu0, delta_f, M):
    mu_cutoff = (np.exp(delta_f)-1)/(M*np.exp(delta_f)-1)
    k = np.floor(power_of_multiple(mu0, mu_cutoff, M))
    return k + 3


def mu_sweep_min(mu0, delta_f, M, f_b, f_a, P_mu, K):
    mu_curr = mu0
    mu_above = M*mu_curr
    rate_up = mutator_sweeprate(mu_curr, mu_above, delta_f, M, f_b, P_mu, K)
    rate_down = antimutator_sweeprate(mu_above, mu_curr, delta_f, M, f_a,
                                      P_mu, K)
    i=0
    if rate_down > 1/100 * rate_up:
        while rate_down > 1/100 * rate_up:
            mu_above = mu_curr
            mu_curr = mu_curr / M
            i = i - 1
            rate_up = mutator_sweeprate(mu_curr, mu_above, delta_f, M, f_b,
                                        P_mu, K)
            rate_down = antimutator_sweeprate(mu_above, mu_curr, delta_f, M,
                                              f_a, P_mu, K)
        return i
    else:
        while rate_down < 1/100 * rate_up:
            mu_curr = mu_above
            mu_above = M*mu_curr
            i = i + 1
            rate_up = mutator_sweeprate(mu_curr, mu_above, delta_f, M, f_b,
                                        P_mu, K)
            rate_down = antimutator_sweeprate(mu_above, mu_curr, delta_f, M,
                                              f_a, P_mu, K)
        return i - 1


def transition_matrix(mu0, delta_f, M, f_b, f_a, P_mu, K):
    l_max = mu_sweep_max(mu0, delta_f, M)
    l_min = mu_sweep_min(mu0, delta_f, M, f_b, f_a, P_mu, K)
    mu_list = mu0*float(M)**np.arange(l_min,l_max)
    if mu_list[-1]>=1:
        raise RuntimeError('These parameter values are outside the range where'
                           ' the invasion-sweep approximation is valid. It is'
                           ' possible for a mutation rate of 1 to sweep the'
                           ' population which will break the assumption of'
                           ' a steady state between sweeps.')
    matrix_size = np.size(mu_list)
    Tm = np.zeros((matrix_size, matrix_size))
    for i in range(matrix_size):
        for j in range(matrix_size):
            if i != j:
                Tij = transition_rate(mu_list[j], mu_list[i], delta_f, M,
                                      f_b, f_a, P_mu, K)
                Tm[i, j] += Tij
                Tm[j, j] -= Tij
    return mu_list, Tm


def steady_state(transition_matrix):
    t_norm = transition_matrix / np.max(transition_matrix)
    U, s, V = np.linalg.svd(t_norm)
    null_vector = V[-1, :] # np svd sorts singular values in descending order
    null_vector = null_vector / np.sum(null_vector)
    null_vector = np.maximum(null_vector, 0)
    proportions = null_vector / np.sum(null_vector)
    return proportions
