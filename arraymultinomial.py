# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 16:40:28 2018

@author: nashere2
"""

import numpy as np


def array_multinomial(N_array, Pis_array, checks=True):
    """
    Draw from multinomial distribution P(x1, x2,..., xi; N, p1, p2, ... pi)
    array wise where N's are from N_array and their respective pi's are from
    Pis_array with matching trailing indices and the leading index is the
    value of i.

    Return an array of xis in the same shape as the array Pis_array.

    The last subarray along the leading axis of Pis_array is ignored if checks
    is False because it is assumed to be equal to 1 minus the rest of that axis
    since probability is conserved. If checks is True then we check to make
    sure all sums over pi's are equal to 1 as required.
    """

    if checks is True:
        if not np.all(Pis_array >= 0):
            raise ValueError('All probabilities must be 0 or positive.')
        total_probability = np.sum(Pis_array, axis=0)
        if not np.all(np.isclose(total_probability, 1., rtol=0, atol=1e-15)):
            raise ValueError('The total probability parameters of a'
                             ' multinomial distribution must sum to 1.')

    if Pis_array.shape[1:] != N_array.shape:
        raise AttributeError('Pis_array must be the shape of N_array plus one '
                             'additional axis in the lead')

    Xis_array = np.zeros_like(Pis_array, dtype='int32')
    N_remain_array = np.copy(N_array)
    prob_remain = np.ones_like(N_array, dtype='float64')
    for i in range(Pis_array.shape[0]-1):
        Xis_array[i, ...] = np.random.binomial(N_remain_array,
                                               Pis_array[i, ...]/prob_remain)
        N_remain_array -= Xis_array[i, ...]
        prob_remain -= Pis_array[i, ...]
    Xis_array[-1, ...] = N_remain_array
    return Xis_array


def approximate_binomial(N, P, size=None):
    N_below_32bit = N * (N <= 2 * 10**9)
    N_above_32bit = N - N_below_32bit
    draw_below = np.random.binomial(N_below_32bit.astype('int32'), P)
    N_broadcast, P_broadcast = np.broadcast_arrays(N_above_32bit, P)
    NP = N_broadcast*P_broadcast
    poisson_approx_segment = NP < 100
    draw_poisson = np.random.poisson(NP*poisson_approx_segment)
    N_normal_approx = N_broadcast * (1 - poisson_approx_segment)
    normal_std = np.sqrt(N_normal_approx * P_broadcast * (1 - P_broadcast))
    draw_normal = np.random.normal(N_normal_approx*P_broadcast, normal_std)
    return (draw_below + draw_poisson + draw_normal).astype('int64')


def array_multinomial_int64(N_array, Pis_array, checks=True):
    """
    Draw from multinomial distribution P(x1, x2,..., xi; N, p1, p2, ... pi)
    array wise where N's are from N_array and their respective pi's are from
    Pis_array with matching trailing indices and the leading index is the
    value of i.

    Return an array of xis in the same shape as the array Pis_array.

    The last subarray along the leading axis of Pis_array is ignored if checks
    is False because it is assumed to be equal to 1 minus the rest of that axis
    since probability is conserved. If checks is True then we check to make
    sure all sums over pi's are equal to 1 as required.
    """

    if checks is True:
        if not np.all(Pis_array >= 0):
            raise ValueError('All probabilities must be 0 or positive.')
        total_probability = np.sum(Pis_array, axis=0)
        if not np.all(np.isclose(total_probability, 1., rtol=0, atol=1e-15)):
            raise ValueError('The total probability parameters of a'
                             ' multinomial distribution must sum to 1.')

    if Pis_array.shape[1:] != N_array.shape:
        raise AttributeError('Pis_array must be the shape of N_array plus one '
                             'additional axis in the lead')

    Xis_array = np.zeros_like(Pis_array, dtype='int64')
    N_remain_array = np.copy(N_array)
    prob_remain = np.ones_like(N_array, dtype='float64')
    for i in range(Pis_array.shape[0]-1):
        Xis_array[i, ...] = approximate_binomial(N_remain_array,
                                                 Pis_array[i, ...]/prob_remain)
        N_remain_array -= Xis_array[i, ...]
        prob_remain -= Pis_array[i, ...]
    Xis_array[-1, ...] = N_remain_array
    return Xis_array
