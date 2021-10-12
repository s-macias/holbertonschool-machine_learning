#!/usr/bin/env python3
""" Module with convolve_grayscale_same function """

import numpy as np
from math import ceil, floor


def convolve_grayscale_same(images, kernel):
    """
    Function that that performs a same convolution
    on grayscale images
    Arguments:
    images is a numpy.ndarray with shape (m, h, w)
    containing multiple grayscale images
    m is the number of images
    h is the height in pixels of the images
    w is the width in pixels of the images
    kernel is a numpy.ndarray with shape (kh, kw)
    containing the kernel for the convolution
    kh is the height of the kernel
    kw is the width of the kernel
    if necessary, the image should be padded with 0’s
    You are only allowed to use two for loops; any other
    loops of any kind are not allowed
    Returns:
    a numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape[0], images.shape[1], images.shape[2]
    kh, kw = kernel.shape[0], kernel.shape[1]

    h_out = max((kh - 1) // 2, kh // 2)
    w_out = max((kw - 1) // 2, kw // 2)
    padding = np.pad(images, ((0, 0),
                              (h_out, h_out),
                              (w_out, w_out)))

    conv_images = np.zeros((m, h, w))

    for i in range(h):
        for j in range(w):
            conv_images[:, i, j] = np.sum(np.multiply(
                padding[:, i:kh+i, j:kw+j], kernel), axis=(1, 2))
    return (conv_images)
