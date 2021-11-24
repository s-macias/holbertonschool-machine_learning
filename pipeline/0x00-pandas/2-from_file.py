#!/usr/bin/env python3
""" Module to create from_file function """

import pandas as pd

def from_file(filename, delimiter):
    """
    Function that loads data from a file as a pd.DataFrame
    Arguments:
    filename is the file to load from
    delimiter is the column separator
    Returns:
    the loaded pd.DataFrame
    """
    return pd.read_csv(filename, sep=delimiter)
