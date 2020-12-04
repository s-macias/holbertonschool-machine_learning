#!/usr/bin/env python3
"""adds two arrays element-wise"""


def add_arrays(arr1, arr2):
    """ adds two arrays arr1 and arr2 element wise """
    if len(arr1) != len(arr2):
        return None
    else:
        new_array = []
        for item in range(len(arr1)):
            new_array.append(arr1[item] + arr2[item])
        return new_array
