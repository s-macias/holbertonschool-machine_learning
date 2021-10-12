#!/usr/bin/env python3
""" Module with convolve_grayscale_valid function """

import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    Function that performs a valid convolution on grayscale images.
    Arguments:
    images is a numpy.ndarray with shape (m, h, w) containing
    multiple grayscale images
    m is the number of images
    h is the height in pixels of the images
    w is the width in pixels of the images
    kernel is a numpy.ndarray with shape (kh, kw) containing
    the kernel for the convolution
    kh is the height of the kernel
    kw is the width of the kernel
    You are only allowed to use two for loops; any other loops
    of any kind are not allowed
    Returns:
    a numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape[0], images.shape[1], images.shape[2]
    kh, kw = kernel.shape[0], kernel.shape[1]

    h_out, w_out = h - kh + 1, h - kw + 1

    conv_images = np.zeros((m, h_out, w_out))

    for i in range(h_out):
        for j in range(w_out):
            conv_images[:, i, j] = np.sum(np.multiply(images[:, i:kh+i, j:kw+j],
                                                 kernel), axis=(1, 2))
    return (conv_images)
