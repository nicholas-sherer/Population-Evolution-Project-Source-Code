# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np


# saves time because it avoid allocations
def multibase_increment_fast(lis, list_bounds):
    i = -1
    while i >= -len(lis):
        if lis[i] + 1 < list_bounds[i]:
            lis[i] = lis[i] + 1
            break
        else:
            lis[i] = 0
            i = i - 1


def int_to_indices(integer, dimensions):
    dimensions = np.array(dimensions)
    if integer > np.product(dimensions):
        raise ValueError('''The number of entries in an array of these
                         dimensions is less than integer''')
    remaining = integer
    indices = ()
    for i in range(-1, -dimensions.size-1, -1):
        base = dimensions[i]
        digit = remaining % base
        remaining = (remaining - digit) // base
        indices = (digit,) + indices
    return indices


def subarray_multislice(array_ndim, axes, indices):
    indices = np.array(indices)
    colon = slice(None, None, None)
    multislice = ()
    for i in range(array_ndim):
        if i in axes:
            multislice = multislice + (indices[np.where(axes == i)[0][0]],)
        else:
            multislice = multislice + (colon,)
    return multislice


def subarray_view(array, axes, indices, checks=True):
    '''
    Return view of subarray of input array with given axes fixed at
    corresponding indices.'''
    if checks:
        # Coerce the inputs into flat numpy arrays to allow for easy handling
        # of a variety of input types
        axes = np.atleast_1d(np.array(axes)).flatten()
        indices = np.atleast_1d(np.array(indices)).flatten()
        check_axes_access(axes, array.ndim)
        convert_axes_to_positive(axes, array.ndim)
        if axes.shape != indices.shape:
            raise ValueError('''axes and indices must have matching shapes or
                             both be integers''')
    return array[subarray_multislice(array.ndim, axes, indices)]


def subrange_view(array, starts, ends, steps=None, checks=True):
    if checks:
        # Coerce the inputs into flat numpy arrays to allow for easy handling
        # of a variety of input types
        starts = np.atleast_1d(np.array(starts)).flatten()
        ends = np.atleast_1d(np.array(ends)).flatten()
        if steps is not None:
            steps = np.atleast_1d(np.array(steps)).flatten()
        # Check number of array axes matches up with starts and ends
        if (array.ndim != starts.size) or (array.ndim != ends.size):
            raise ValueError('''the size of starts and ends must equal the
                             number of array dimensions''')
    multislice = ()
    # If steps is None, default to step size of 1
    if steps is None:
        for i in range(array.ndim):
            multislice = multislice + (slice(starts[i], ends[i], 1),)
    else:
        for i in range(array.ndim):
            multislice = multislice + (slice(starts[i], ends[i], steps[i]),)
    return array[multislice]


def check_axes_access(axes, array_ndim):
    if np.max(axes) >= array_ndim or np.min(axes) < -array_ndim:
            raise IndexError('too many indices for array')


def convert_axes_to_positive(axes, array_ndim):
    for index, element in enumerate(axes):
            if element < 0:
                axes[index] = element + array_ndim


def check_stencil_shape(array_ndim, axes, summed_axes_shape, stencil):
    correct_stencil_shape = np.hstack([np.array(summed_axes_shape),
                                       np.array(array_ndim - len(axes))])
    if not np.all(np.array(stencil.shape) == correct_stencil_shape):
            raise ValueError('''The shape of the stencil must match the big
                             array and axes appropriately''')


def stenciled_sum(big_array, axes, stencil, checks=True):
    if checks:
        axes = np.atleast_1d(np.array(axes)).flatten()
        check_axes_access(axes, big_array.ndim)
        convert_axes_to_positive(axes, big_array.ndim)
        # if we're summing across every axis, then just call np.sum
        # (this avoids complicating rest of code for simple special case)
        if big_array.ndim == len(axes):
            return np.sum(big_array)
        summed_axes_shape = np.array(big_array.shape)[axes]
        check_stencil_shape(big_array.ndim, axes, summed_axes_shape, stencil)

    # make array of the axes we're preserving
    not_axes = [i for i in range(big_array.ndim) if i not in axes]
    # tuple of all but last stencil axis
    ablsa = tuple(range(stencil.ndim-1))
    subarray_shape = np.array(big_array.shape)[not_axes]
    return_array_shape = subarray_shape + \
        np.amax(stencil, axis=ablsa) - np.amin(stencil, axis=ablsa)

    # left zero the stencil
    stencil = stencil - np.amin(stencil, axis=ablsa)

    # perform the stenciled summation
    return_array = np.zeros(return_array_shape, dtype=big_array.dtype)
    iter_bounds = stencil.shape[:-1]
    final_loop = np.product(np.array(iter_bounds))
    index = np.zeros(len(iter_bounds), dtype='int_')

    for i in range(final_loop):
        starts = stencil[tuple(index)]
        ends = stencil[tuple(index)] + subarray_shape
        chunk_to_increase = subrange_view(return_array, starts, ends,
                                          checks=checks)
        chunk_to_increase[:] += subarray_view(big_array, axes, index,
                                              checks=checks)
        multibase_increment_fast(index, iter_bounds)
    return return_array


class fixedStencilSum(object):

    def __init__(self, array_ndim, axes_summed_over, summed_axes_shape,
                 stencil):
        axes = np.atleast_1d(np.array(axes_summed_over)).flatten()
        # check that inputs are compatible
        check_axes_access(axes, array_ndim)
        convert_axes_to_positive(axes, array_ndim)
        try:
            check_stencil_shape(array_ndim, axes, summed_axes_shape, stencil)
        except ValueError as e:
            # For the trivial case where we collapse the array to one number,
            # just monkeypatch the only method of the class to the simple
            # implementation. Parameters become irrelevant.
            if array_ndim == len(axes_summed_over):
                self.stenciled_sum = np.sum
            else:
                raise e

        self.array_ndim = array_ndim
        self.axes = axes
        self.not_axes = [i for i in range(array_ndim) if i not in axes]
        self.summed_axes_shape = summed_axes_shape
        # left zero the stencil, ablsa is a tuple of indices into
        # "all but last stencil axis"
        ablsa = tuple(range(stencil.ndim-1))
        stencil = stencil - np.amin(stencil, axis=ablsa)
        self.stencil = stencil
        self.input_expand = np.amax(stencil, axis=ablsa) - \
            np.amin(stencil, axis=ablsa)
        self.iter_bounds = stencil.shape[:-1]
        final_loop = np.product(np.array(self.iter_bounds))

        self.stencil_loop_indices = [int_to_indices(i, self.iter_bounds)
                                     for i in range(final_loop)]
        self.multislices = [subarray_multislice(self.array_ndim, self.axes,
                                                indices) for indices in
                            self.stencil_loop_indices]

    def stenciled_sum(self, big_array):
        subarray_shape = np.array(big_array.shape)[self.not_axes]
        return_array_shape = subarray_shape + self.input_expand
        return_array = np.zeros(return_array_shape, dtype=big_array.dtype)
        for indices, multislice in zip(self.stencil_loop_indices,
                                       self.multislices):
            starts = self.stencil[indices]
            ends = self.stencil[indices] + subarray_shape
            chunk_to_increase = subrange_view(return_array, starts, ends,
                                              checks=False)
            chunk_to_increase[:] += big_array[multislice]
        return return_array
