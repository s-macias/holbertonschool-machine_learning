#!/usr/bin/env python3
""" Module to create from_numpy(array) function """

import pandas as pd

def from_numpy(array):
    """
    Function that creates a pd.DataFrame from a np.ndarray:
    Arguments:
    array is the np.ndarray from which you should create the pd.DataFrame
    The columns of the pd.DataFrame should be labeled in alphabetical order
    and capitalized. There will not be more than 26 columns.
    Returns:
    the newly created pd.DataFrame
    """
    labels = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    cols = array.shape[1]
    return pd.DataFrame(array, columns=labels[:cols])
