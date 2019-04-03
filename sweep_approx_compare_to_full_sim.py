# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 17:27:37 2019

@author: Nicholas Sherer
"""

import numpy as np
import populationevolution_v5 as popev
import raggedtoregular as r2r
import SweepApproximationFunctions as SAF


def compareNeq_Ntrue(mu_min, delta_f, M, P_mu, K, t):
    Neq = SAF.findNeq(mu_min, delta_f, M, P_mu, K)
    N_start = Neq.astype('int64')
    rounding_error = K - np.sum(N_start)
    N_start[0,0] = N_start[0,0] + rounding_error
    pop = popev.Population(0, mu_min, N_start, delta_f, M, 0, 0, P_mu, K)
    mu_list = pop.mutation_list
    delta_Ns = [diffNeq_Npop(mu_list, Neq, pop.mutation_list,
                             pop.population_distribution)]
    for i in range(t):
        pop.update()
        delta_Ns.append(diffNeq_Npop(mu_list, Neq, pop.mutation_list,
                                     pop.population_distribution))
    delta_Ns = r2r.ragged_to_regular(delta_Ns)
    new_shape = (delta_Ns.shape[1], delta_Ns.shape[2])
    Neq_rs = np.zeros(new_shape, dtype=Neq.dtype)
    Neq_rs[:Neq.shape[0],:Neq.shape[1]] = Neq
    mu_list_rs = np.minimum(np.geomspace(mu_min,
                                         mu_min*M**(new_shape[1]-1),
                                         new_shape[1]),1)
    f_list = np.linspace(0, -delta_f*new_shape[0], new_shape[0],
                         endpoint=False)
    return mu_list_rs, f_list, Neq_rs, delta_Ns


def diffNeq_Npop(mu_list, Neq, pop_mut, pop_dist):
    fs, mus = np.maximum(pop_dist.shape, Neq.shape)
    M = mu_list[1]/mu_list[0]
    Neq_padded = np.zeros((fs,mus),dtype=Neq.dtype)
    Neq_padded[:Neq.shape[0],:Neq.shape[1]]=Neq
    mu_list_padded = np.minimum(mu_list[0]*M**np.arange(mus),1)
    pop_dist_padded = np.zeros((fs,mus),dtype=pop_dist.dtype)
    pop_dist_padded[:pop_dist.shape[0],:pop_dist.shape[1]]=pop_dist
    pop_mut_padded = np.minimum(pop_mut[0]*M**np.arange(mus),1)
    if not np.allclose(pop_mut_padded, mu_list_padded):
        print('eq_mu_list:', mu_list_padded)
        print('pop_mu_list:', pop_mut_padded)
        raise RuntimeError('the mutation rates of Neq and the population being'
                           'tested do not match.')
    return pop_dist_padded - Neq_padded


def mean_probability_error(delta_Ns, K):
    '''
    Calculate the sum of the absolute value of delta_Ns over time then divide
    by 2.'''
    # The first axis of the array delta_Ns is the time axis so taking
    # the cumulative sum and dividing by the array ts give us delta_N
    # averaged over time for increasing values of time. The second two axes
    # are the fitness and mutation rate axes so taking the absolute value,
    # summing over those and dividing by 2*K gives us a normalized measure
    # of the deviation of N from the approximate equilibrium. We divide by 2
    # because that way if there is no overlap in the distributions the measure
    # will equal 1.
    ts = np.arange(1,delta_Ns.shape[0]+1).reshape((-1,1,1))
    return np.sum(np.abs(np.cumsum(delta_Ns,axis=0)/ts), axis=(1,2))/(2*K)


def delta_over_std(Neq, delta_N):
    K = np.sum(Neq)
    std = np.sqrt(Neq*(1-Neq/K))
    return delta_N/std


def invader_Neq(mu_min, delta_f, M, P_mu, K, inv_f_step, inv_mu_step):
    s = SAF.effective_fitness_difference(0, delta_f*inv_f_step, mu_min,
                                         mu_min*float(M)**inv_mu_step)
    if inv_f_step < 0:
        raise ValueError('Invading mutants must have a fitness equal to or'
                         ' above the maximum fitness in the invaded'
                         ' population.')
    elif inv_f_step == 0:
        if inv_mu_step > -1:
            raise ValueError('Invading mutants at the same fitness as the'
                             ' invaded population must have a lower mutation'
                             ' rate.')
    else:
        if SAF.fixation_probability(K, s) == 0:
            raise ValueError('Invading mutants must have an effective increase'
                             ' in fitness or be effectively neutral. Try '
                             'increasing the fitness or decreasing the'
                             ' mutation rate of the invader.')
    Neq = SAF.findNeq(mu_min, delta_f, M, P_mu, K)
    N_start = Neq.astype('int64')
    rounding_error = K - np.sum(N_start)
    N_start[0,0] = N_start[0,0] + rounding_error - 1
    vertical_pad = (inv_f_step, 0)
    if inv_mu_step < 0:
        horizontal_pad = (np.abs(inv_mu_step),0)
    elif inv_mu_step >= N_start.shape[1]:
        horizontal_pad = (0, inv_mu_step - N_start.shape[1])
    else:
        horizontal_pad = (0,0)
    N_start = np.pad(N_start, (vertical_pad, horizontal_pad),mode='constant')
    N_start[0,np.maximum(0,inv_mu_step)] = 1
    return N_start


def invasion(mu_min, delta_f, M, P_mu, K, inv_f_step, inv_mu_step,
             max_steps=10**6):
    N_start = invader_Neq(mu_min, delta_f, M, P_mu, K, inv_f_step, inv_mu_step)
    mu_inv = mu_min*float(M)**inv_mu_step
    f_inv = delta_f*inv_f_step
    if inv_mu_step < 0:
        mu_min2 = mu_inv
    else:
        mu_min2 = mu_min
    pop = popev.Population(f_inv, mu_min2, N_start, delta_f, M, 0, 0, P_mu,
                           K)
    threshold = .5*SAF.findNeq(mu_inv, delta_f, M, P_mu, K)[0,0]
    for i in range(1,max_steps):
        pop.update()
        if pop(f_inv, mu_inv) == 0:
            return False, i
        if pop(f_inv, mu_inv) >= threshold:
            return True, i
    raise RuntimeError('The invading mutant failed to either fix or'
                       ' go extinct before the maximum number of allowable'
                       ' function evaluations was reached.')


def estimate_fix_prob(mu_min, delta_f, M, P_mu, K, inv_f_step, inv_mu_step):
    s = SAF.effective_fitness_difference(0, delta_f*inv_f_step, mu_min,
                                         mu_min*float(M)**inv_mu_step)
    exp_fix_prob = SAF.fixation_probability(K, s)
    if exp_fix_prob < 10**-5:
        raise RuntimeError('Empirically estimating the fixation probability'
                           ' for such an unlikely event is a waste of time.')
    test_count = int(np.maximum(100*(1-exp_fix_prob)/exp_fix_prob,100))
    fixations = 0
    extinction_times = []
    fixation_times = []
    for i in range(test_count):
        invader_survival, time = invasion(mu_min, delta_f, M, P_mu, K,
                                          inv_f_step, inv_mu_step)
        if invader_survival:
            fixations = fixations + 1
            fixation_times.append(time)
        else:
            extinction_times.append(time)
    return test_count, fixations, fixation_times, extinction_times