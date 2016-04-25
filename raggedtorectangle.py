# -*- coding: utf-8 -*-
"""
Created on Tue Feb 09 13:47:16 2016

@author: Nicholas Sherer
"""

from __future__ import division

import numpy as np


def raggedTo3DRectangle(array_list, pad_constant=0):
    """
    This function takes a list of 2d numpy arrays and turns them into a 3d
    numpy array padded out by a number of the users choice.
    """
    join_length = len(array_list)
    x = np.zeros(join_length)
    y = np.zeros(join_length)
    for i in range(join_length):
        x[i] = np.shape(array_list[i])[0]
        y[i] = np.shape(array_list[i])[1]
    max_x = max(x)
    max_y = max(y)
    padded_rectangle = np.ones((max_x, max_y, join_length)) * pad_constant
    for i in range(join_length):
        padded_rectangle[0:x[i], 0:y[i], i] = array_list[i]
    return padded_rectangle
