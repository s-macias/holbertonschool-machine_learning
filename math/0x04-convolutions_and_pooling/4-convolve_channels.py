#!/usr/bin/env python3
""" Module with convolve_channels function """

import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
    Function that performs a convolution on images with channels
    Arguments:
    images is a numpy.ndarray with shape (m, h, w, c)
    containing multiple grayscale images
    m is the number of images
    h is the height in pixels of the images
    w is the width in pixels of the images
    c is the number of channels in the image
    kernel is a numpy.ndarray with shape (kh, kw, c)
    containing the kernel for the convolution
    kh is the height of the kernel
    kw is the width of the kernel
    padding is either a tuple of (ph, pw), ‘same’, or ‘valid’
    if ‘same’, performs a same convolution
    if ‘valid’, performs a valid convolution
    if a tuple:
        ph is the padding for the height of the image
        pw is the padding for the width of the image
        the image should be padded with 0’s
        stride is a tuple of (sh, sw)
    sh is the stride for the height of the image
    sw is the stride for the width of the image
    You are only allowed to use two for loops; any other loops
    of any kind are not allowed Hint: loop over i and j
    Returns:
    a numpy.ndarray containing the convolved images
    """
    m, h, w, c = images.shape
    kh, kw, _ = kernel.shape
    sh, sw = stride

    if padding == 'same':
        ph = int(((w - 1) * sw + kw - w) / 2) + 1
        pw = int(((h - 1) * sh + kh - h) / 2) + 1
    if padding == 'valid':
        ph = 0
        pw = 0
    if type(padding) is tuple:
        ph, pw = padding

    h_out = (h - kh + 2 * ph) // sh + 1
    w_out = (w - kw + 2 * pw) // sw + 1

    conv_images = np.zeros((m, h_out, w_out))
    pad_images = np.pad(images, [(0, 0), (ph, ph), (pw, pw), (0, 0)])

    for i in range(h_out):
        for j in range(w_out):
            conv_images[:, i, j] = np.sum(np.multiply(
                pad_images[:, i * sh: (i*sh) + kh, j*sw:
                           (j*sw) + kw], kernel), axis=(1, 2, 3))

    return conv_images
