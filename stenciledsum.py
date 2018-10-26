# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np


def subarray_multislice(array_ndim, fixed_axes, indices):
    '''
    Return tuple of slices that if indexed into an array with given dimensions
    will return subarray with the axes in axes fixed at given indices
    '''
    indices = np.array(indices)
    colon = slice(None, None, None)
    multislice = ()
    for i in range(array_ndim):
        if i in fixed_axes:
            multislice = multislice + \
                (indices[np.where(fixed_axes == i)[0][0]],)
        else:
            multislice = multislice + (colon,)
    return multislice


def subarray_view(array, fixed_axes, indices, checks=True):
    '''
    Return view of subarray of input array with fixed_axes at
    corresponding indices.'''
    if checks:
        # Coerce the inputs into flat numpy arrays to allow for easy handling
        # of a variety of input types
        fixed_axes = np.atleast_1d(np.array(fixed_axes)).flatten()
        indices = np.atleast_1d(np.array(indices)).flatten()
        check_axes_access(fixed_axes, array.ndim)
        convert_axes_to_positive(fixed_axes, array.ndim)
        if fixed_axes.shape != indices.shape:
            raise ValueError('axes and indices must have matching shapes or'
                             ' both be integers')
    return array[subarray_multislice(array.ndim, fixed_axes, indices)]


def subrange_view(array, starts, ends, steps=None, checks=True):
    '''
    Return view of array with each axes indexed between starts and ends.
    '''
    if checks:
        # Coerce the inputs into flat numpy arrays to allow for easy handling
        # of a variety of input types
        starts = np.atleast_1d(np.array(starts)).flatten()
        ends = np.atleast_1d(np.array(ends)).flatten()
        if steps is not None:
            steps = np.atleast_1d(np.array(steps)).flatten()
        # Check number of array axes matches up with starts and ends
        if (array.ndim != starts.size) or (array.ndim != ends.size):
            raise ValueError('the size of starts and ends must equal the '
                             'number of array dimensions')
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


# regular numpy scheme for which positive index a negative index corresponds to
def convert_axes_to_positive(axes, array_ndim):
    for index, element in enumerate(axes):
            if element < 0:
                axes[index] = element + array_ndim


def correct_stencil_shape(array_ndim, axes, summed_axes_shape):
    return np.hstack([np.array(summed_axes_shape),
                     np.array(array_ndim - len(axes))])


def check_stencil_shape(array_ndim, axes, summed_axes_shape, stencil):
    if not np.all(np.array(stencil.shape) ==
                  correct_stencil_shape(array_ndim, axes, summed_axes_shape)):
            raise ValueError('The shape of the stencil must match the big'
                             ' array and axes appropriately')


def stenciled_sum(array, summed_axes, stencil):
    summed_axes = np.atleast_1d(np.array(summed_axes))
    summed_axes_shape = np.array(array.shape)[summed_axes]
    fixed_stencil_summer = fixedStencilSum(array.ndim, summed_axes,
                                           summed_axes_shape, stencil)
    return fixed_stencil_summer.stenciled_sum(array)


class fixedStencilSum(object):

    def __init__(self, array_ndim, axes_summed_over, summed_axes_shape,
                 stencil):
        axes = np.atleast_1d(np.array(axes_summed_over)).flatten()
        # check that inputs are compatible
        check_axes_access(axes, array_ndim)
        convert_axes_to_positive(axes, array_ndim)
        check_stencil_shape(array_ndim, axes, summed_axes_shape, stencil)
        # handle a trivial case where we sum the entire array into one number
        if array_ndim == len(axes):
            self.stenciled_sum = np.sum

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
        self.stencil_loop_indices = [i for i in np.ndindex(stencil.shape[:-1])]
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

    def __eq__(self, other):
        if isinstance(other, fixedStencilSum):
            is_equal = True
            for attribute in self.__dict__:
                is_equal = (np.all(np.equal(getattr(self, attribute),
                            getattr(other, attribute)))) and is_equal
            return is_equal
        else:
            return NotImplemented
