# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 12:26:48 2018

@author: nashere2
"""

import numpy as np
import stenciledsum as stsum


def unstenciled_sum(big_array, axes, checks=True):
    '''Sum big_array across axes.

    Should return a result equivalent to calling
    np.sum(big_array, axis = tuple(axes)).'''
    correct_stencil_shape = np.hstack([np.array(big_array.shape)[axes],
                                      np.array(big_array.ndim - len(axes))])
    stencil = np.zeros(correct_stencil_shape, dtype='int_')
    return stsum.stenciled_sum(big_array, axes, stencil)


def test_equality_summation(big_array, axes, verbose=False):
    '''Test equality of unstenciled sum and np.sum with same arguments.'''
    if verbose:
        print('summation using unstenciled_sum is',
              unstenciled_sum(big_array, axes))
        print('summation using np.sum is',
              np.sum(big_array, axis=tuple(axes)))
    return np.all(np.sum(big_array, axis=tuple(axes)) ==
                  unstenciled_sum(big_array, axes))


def random_eq_test(verbose=False):
    '''Randomized testing of stenciled sum.'''
    ndim = np.random.randint(2, 7)
    shape = np.random.randint(2, 7, size=ndim)
    n_axes_sum = np.random.randint(1, ndim+1)
    axes_sum = np.random.choice(np.arange(n_axes_sum),
                                size=n_axes_sum, replace=False)
    array = np.random.randint(100, size=shape)
    if verbose:
        print('shape is', shape)
        print('axes summed over is', axes_sum)
        print('array is', array)
    try:
        success = test_equality_summation(array, axes_sum, verbose)
    except IndexError:
        success = False
    return success, array, axes_sum
