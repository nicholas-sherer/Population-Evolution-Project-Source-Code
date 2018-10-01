# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 14:27:19 2018

@author: nashere2
"""

import pytest
import numpy as np
import scipy.stats as spstats
import arraymultinomial as am


def check(N, Pis):
    if not np.all(Pis >= 0):
        raise ValueError('All probabilities must be 0 or positive.')
    total_probability = np.sum(Pis, axis=0)
    if not np.all(np.isclose(total_probability, 1., rtol=0, atol=1e-15)):
        raise ValueError('The total probability parameters of a'
                         ' multinomial distribution must sum to 1.')
    if Pis.shape[1:] != N.shape:
        raise AttributeError('Pis_array must be the shape of N_array plus'
                             'one additional axis in the lead')


def multinomial_mean(N, Pis):
    N = np.array(N)
    check(N, Pis)
    return N*Pis


def multinomial_var(N, Pis):
    N = np.array(N)
    check(N, Pis)
    return N*Pis*(1-Pis)


def multinomial_fourth_moment_about_mean(N, Pis):
    N = np.array(N)
    check(N, Pis)
    return N*Pis*(1-Pis)*(3*Pis**2*(2-N)+3*Pis*(N-2)+1)


def stderr_of_mean(var, sample_size):
    return np.sqrt(var/sample_size)


def stderr_of_var(var, mu4, sample_size):
    return np.sqrt((mu4-(sample_size-3)/(sample_size-1)*var**2)/sample_size)


def multinomial_draw_repeated(N, Pis, sample_size):
    N = np.array(N)
    check(N, Pis)
    draws = [am.array_multinomial(N, Pis, checks=False)
             for i in range(sample_size)]
    return draws


def multinomial_draw_sample_mean(draws):
    return np.mean(np.array(draws), axis=0)


def multinomial_draw_sample_var(draws):
    return np.var(np.array(draws), axis=0)


def multinomial_mean_and_var_errors(N, Pis, sample_size):
    means = multinomial_mean(N, Pis)
    varis = multinomial_var(N, Pis)
    mu4s = multinomial_fourth_moment_about_mean(N, Pis)
    draws = multinomial_draw_repeated(N, Pis, sample_size)
    sample_means = multinomial_draw_sample_mean(draws)
    sample_vars = multinomial_draw_sample_var(draws)
    est_mean_errors = stderr_of_mean(varis, sample_size)
    est_var_errors = stderr_of_var(varis, mu4s, sample_size)
    mean_errors = sample_means - means
    var_errors = sample_vars - varis
    return mean_errors / est_mean_errors, var_errors / est_var_errors


def test_sample_means_and_var_distribution(N, Pis, sample_size):
    x, y = multinomial_mean_and_var_errors(N, Pis, sample_size)
    x_tstat = spstats.shapiro(x)[0]
    y_tstat = spstats.shapiro(y)[0]
    assert min(x_tstat, y_tstat) >= .98
    