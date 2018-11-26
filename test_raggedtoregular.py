# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 16:24:41 2018

@author: nashere2
"""

import pytest
import numpy as np
import raggedtoregular as r2r


randint = np.random.randint


def random_list_of_arrays(list_length, array_ndim):
    return [randint(1, 11, size=randint(1, 10, array_ndim))
            for i in range(list_length)]


@pytest.mark.parametrize("array_list",
                         [random_list_of_arrays(randint(1, 100), randint(1, 6))
                          for j in range(1000)])
def test_ragged_to_regular_to_ragged_is_identity(array_list):
    array_regular = r2r.ragged_to_regular(array_list)
    array_ragged = r2r.regular_to_ragged(array_regular)
    equalities = [np.all(orig == new) for orig, new in
                  zip(array_list, array_ragged)]
    isequal = True
    for test in equalities:
        isequal = isequal and test
    assert(isequal)
