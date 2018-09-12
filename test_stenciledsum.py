# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 12:26:48 2018

@author: nashere2
"""

import pytest
import numpy as np
import stenciledsum as stsum


def random_array(ndim_max, shape_max):
    ndim = np.random.randint(2, ndim_max)
    shape = np.random.randint(2, shape_max, size=ndim)
    return np.random.randint(100, size=shape)


def random_axes(array):
    n_axes = np.random.randint(1, array.ndim+1)
    axes = np.random.choice(np.arange(n_axes), size=n_axes, replace=False)
    return axes


def random_stencil(array, axes_to_sum, min_shift, max_shift):
    axes_shape = np.array(array.shape)[np.array(axes_to_sum)]
    stencil_shape = stsum.correct_stencil_shape(array.ndim, axes_to_sum,
                                                axes_shape)
    return np.random.randint(min_shift, max_shift, size=stencil_shape)


def random_array_and_axes_to_sum(ndim_max, shape_max):
    array = random_array(ndim_max, shape_max)
    axes_to_sum = random_axes(array)
    return array, axes_to_sum


def unstenciled_sum(big_array, axes):
    '''Sum big_array across axes.

    Should return a result equivalent to calling
    np.sum(big_array, axis = tuple(axes)).'''
    correct_stencil_shape = np.hstack([np.array(big_array.shape)[axes],
                                      np.array(big_array.ndim - len(axes))])
    stencil = np.zeros(correct_stencil_shape, dtype='int_')
    return stsum.stenciled_sum(big_array, axes, stencil)


@pytest.mark.parametrize("big_array,axes", [random_array_and_axes_to_sum(7, 7)
                                            for i in range(1000)])
def test_equality_of_summation(big_array, axes, verbose=False):
    '''Test equality of unstenciled sum and np.sum with same arguments.'''
    if verbose:
        print('summation using unstenciled_sum is',
              unstenciled_sum(big_array, axes))
        print('summation using np.sum is',
              np.sum(big_array, axis=tuple(axes)))
    assert np.all(np.sum(big_array, axis=tuple(axes)) ==
                  unstenciled_sum(big_array, axes))


def random_complete_reduction(big_array, verbose=False):
    axes_summed = []
    stencils = []
    reduced_arrays = []
    remaining_array = big_array
    while remaining_array.size > 1:
        axes_to_sum = random_axes(remaining_array)
        axes_summed.append(axes_to_sum)
        stencil = random_stencil(remaining_array, axes_to_sum, -4, 4)
        stencils.append(stencil)
        if verbose is True:
            print(remaining_array)
            print(axes_to_sum)
            print(stencil)
        remaining_array = stsum.stenciled_sum(remaining_array, axes_to_sum,
                                              stencil)
        reduced_arrays.append(remaining_array)
    return axes_summed, stencils, reduced_arrays


@pytest.mark.parametrize("array", [random_array(7, 7) for i in range(100)])
def test_complete_reduction(array):
    assert random_complete_reduction(array)[2][-1] == np.sum(array)

pytest.main()
