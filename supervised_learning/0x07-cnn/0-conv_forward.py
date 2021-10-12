#!/usr/bin/env python3

import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    Function that performs forward propagation over a convolutional layer
    of a neural network:
    Arguments:
    A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
    containing the output of the previous layer
    m is the number of examples
    h_prev is the height of the previous layer
    w_prev is the width of the previous layer
    c_prev is the number of channels in the previous layer
    W is a numpy.ndarray of shape (kh, kw, c_prev, c_new) containing
    the kernels for the convolution
    kh is the filter height
    kw is the filter width
    c_prev is the number of channels in the previous layer
    c_new is the number of channels in the output
    b is a numpy.ndarray of shape (1, 1, 1, c_new) containing the
    biases applied to the convolution
    activation is an activation function applied to the convolution
    padding is a string that is either same or valid, indicating
    the type of padding used
    stride is a tuple of (sh, sw) with strides for the convolution
    sh is the stride for the height
    sw is the stride for the width
    you may import numpy as np
    Returns:
    the output of the convolutional layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride
    
    if padding == 'same':
        ph = int(((h_prev - 1) * sh - h_prev + kh) / 2)
        pw = int(((w_prev - 1) * sw - w_prev + kw) / 2)
    else:
        pw, ph = 0, 0

    h_out = (h_prev - kh + 2 * ph) // sh + 1
    w_out = (w_prev - kw + 2 * pw) // sw + 1

    convolved = np.zeros((m, h_out, w_out, c_new))

    padded_A = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)))
    
    for i in range(h_out):
        for j in range(w_out):
            for k in range(c_prev):
                convolved[:, i, j, k] = np.sum(np.multiply(
                    padded_A[:, i*sh: i*sh + kh, j * sw : j + kw, :],
                     W[:, :, :, k]), axis=(1, 2, 3))
    return activation(convolved + b)
