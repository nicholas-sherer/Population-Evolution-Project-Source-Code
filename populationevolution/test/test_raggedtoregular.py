# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 16:24:41 2018

@author: nashere2
"""

import pytest
import numpy as np
from .. import raggedtoregular as r2r


randint = np.random.randint


def random_list_of_arrays(list_length, array_ndim):
    return [randint(1, 11, size=randint(1, 10, array_ndim))
            for i in range(list_length)]


@pytest.mark.parametrize("array_list",
                         [random_list_of_arrays(randint(1, 100), randint(1, 4))
                          for i in range(100)])
def test_ragged_to_regular_to_ragged_is_identity(array_list):
    '''
    Test that for lists of arrays without zeros applying ragged_to_regular
    followed by regular_to_ragged is the identity.

    Note that this won't hold in general for lists of arrays that may contain
    zero.
    '''
    array_regular = r2r.ragged_to_regular(array_list)
    array_ragged = r2r.regular_to_ragged(array_regular)
    equalities = [np.all(orig == new) for orig, new in
                  zip(array_list, array_ragged)]
    isequal = True
    for test in equalities:
        isequal = isequal and test
    assert(isequal)


def random_array_array(array_ndim):
    shape = randint(1, 10, array_ndim)
    array_full = randint(1, 10, size=shape)
    colon = slice(None, None, None)
    for i in range(1, array_full.shape[0]):
        subarray = array_full[i]
        for j in range(subarray.ndim):
            if subarray.shape[j] != 1:
                multislice = ()
                for k in range(subarray.ndim):
                    if k == j:
                        removal_start = randint(1, subarray.shape[j])
                        multislice = multislice + \
                            (slice(removal_start, None, 1),)
                    else:
                        multislice = multislice + (colon,)
                subarray[multislice] = 0
    return array_full


@pytest.mark.parametrize("array_array",
                         [random_array_array(randint(1, 4))
                          for i in range(100)])
def test_regular_to_ragged_to_regular_is_identity(array_array):
    '''
    Test that for an array where no subarray along the first axes has a leading
    face of all zeros (i.e. a subarray of the subarray indexed by 0) applying
    regular_to_ragged followed by ragged_to_regular is the identity.

    Note that this won't hold in general for arrays where the leading face of a
    subarray may be zero.
    '''
    array_ragged = r2r.regular_to_ragged(array_array)
    array_regular = r2r.ragged_to_regular(array_ragged)
    equalities = [np.all(orig == new) for orig, new in
                  zip(array_array, array_regular)]
    isequal = True
    for test in equalities:
        isequal = isequal and test
    assert(isequal)

#pytest.main()
