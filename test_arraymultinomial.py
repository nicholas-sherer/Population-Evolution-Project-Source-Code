# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 14:27:19 2018

@author: nashere2
"""

import pytest
import numpy as np
import scipy.stats as spstats
import arraymultinomial as am


def multinomial_mean(N, Pis):
    N = np.array(N)
    am.check(N, Pis)
    return N*Pis


def multinomial_var(N, Pis):
    N = np.array(N)
    am.check(N, Pis)
    return N*Pis*(1-Pis)


def multinomial_fourth_moment_about_mean(N, Pis):
    N = np.array(N)
    am.check(N, Pis)
    return N*Pis*(1-Pis)*(3*Pis**2*(2-N)+3*Pis*(N-2)+1)


def stderr_of_mean(var, sample_size):
    return np.sqrt(var/sample_size)


def stderr_of_var(var, mu4, sample_size):
    return np.sqrt((mu4-(sample_size-3)/(sample_size-1)*var**2)/sample_size)


def multinomial_draw_repeated(N, Pis, sample_size,
                              multi=am.array_multinomial):
    N = np.array(N)
    am.check(N, Pis)
    return [multi(N, Pis, checks=False) for i in range(sample_size)]


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


def random_N_and_Pis_arrayint32(Pis_0max):
    ndim = np.random.randint(1, 3)
    shape = tuple(np.random.randint(2, 10, size=ndim))
    N = np.random.randint(100, 10000, size=shape, dtype='int32')
    probabilities_length = np.random.randint(2, Pis_0max)
    Pis_shape = (probabilities_length,) + shape
    Pis = np.zeros(Pis_shape)
    for i in range(probabilities_length-1):
        Pis[i, ...] = np.random.uniform(0, 1-np.sum(Pis, axis=0))
    Pis[probabilities_length-1, ...] = 1 - np.sum(Pis, axis=0)
    return N, Pis


def random_N_and_Pis_arrayint64(Pis_0max):
    ndim = np.random.randint(1, 3)
    shape = tuple(np.random.randint(2, 10, size=ndim))
    N = np.int64(10**np.random.randint(0, 9, size=shape))*10**7
    probabilities_length = np.random.randint(2, Pis_0max)
    Pis_shape = (probabilities_length,) + shape
    Pis = np.zeros(Pis_shape)
    for i in range(probabilities_length-1):
        Pis[i, ...] = np.random.uniform(0, 1-np.sum(Pis, axis=0))
    Pis[probabilities_length-1, ...] = 1 - np.sum(Pis, axis=0)
    return N, Pis


def random_N_and_Pis_scalarint64(Pis_0max):
    N = np.int64(10**np.random.randint(0, 9))*10**7
    probabilities_length = np.random.randint(2, Pis_0max)
    Pis = np.zeros(probabilities_length)
    for i in range(probabilities_length-1):
        Pis[i] = np.random.uniform(0, 1-np.sum(Pis, axis=0))
    Pis[probabilities_length-1] = 1 - np.sum(Pis)
    return N, Pis

@pytest.mark.parametrize("N,Pis,sample_size",
                         [random_N_and_Pis(10) + (1000,) for i in range(100)])
def test_sample_means_and_var_distribution(N, Pis, sample_size):
    x, y = multinomial_mean_and_var_errors(N, Pis, sample_size)
    x_pvalue = spstats.shapiro(x)[1]
    y_pvalue = spstats.shapiro(y)[1]
    assert min(x_pvalue, y_pvalue) >= .05


@pytest.mark.parametrize("N,Pis", [random_N_and_Pis(10) for i in range(100)])
def test_draws_sum_to_N(N, Pis):
    draw = am.array_multinomial(N, Pis)
    assert np.all(np.sum(draw, axis=0) == N)


pytest.main()
