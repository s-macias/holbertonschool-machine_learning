#!/usr/bin/env python3
""" Module with pool function """

import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    Function that performs pooling on images
    Arguments:
    images is a numpy.ndarray with shape (m, h, w, c)
    containing multiple grayscale images
    m is the number of images
    h is the height in pixels of the images
    w is the width in pixels of the images
    c is the number of channels in the image
    kernel is a numpy.ndarray with shape (kh, kw)
    containing the kernel for the convolution
    kh is the height of the kernel
    kw is the width of the kernel
    stride is a tuple of (sh, sw)
    sh is the stride for the height of the image
    sw is the stride for the width of the image
    mode indicates the type of pooling
    max indicates max pooling
    avg indicates average pooling
    You are only allowed to use two for loops; any other loops
    of any kind are not allowed Hint: loop over i and j
    Returns:
    a numpy.ndarray containing the pooled images
    """
    m, h, w, c = images.shape
    kh, kw = kernel_shape[0], kernel_shape[1]
    sh, sw = stride

    h_out = (h - kh) // sh + 1
    w_out = (w - kw) // sw + 1

    pool_images = np.zeros((m, h_out, w_out, c))

    for i in range(h_out):
        for j in range(w_out):
            if mode == 'max':

                pool_images[:, i, j] = np.max(images[:, i * sh: (i*sh) + kh, j*sw:
                           (j*sw) + kw],  axis=(1, 2))
            if mode == 'avg':
                pool_images[:, i, j] = np.mean(images[:, i * sh: (i*sh) + kh, j*sw:
                           (j*sw) + kw],  axis=(1, 2))


    return pool_images
