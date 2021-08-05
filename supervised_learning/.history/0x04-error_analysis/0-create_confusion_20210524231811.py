#!/usr/bin/env python3
""" Module that creates a confusion matrix """

def create_confusion_matrix(labels, logits):
    """Function that creates confusion matrix

    Args:
        labels ([numpy.ndarray]): array with shape (m, classes)
            containing the correct labels for each data point
m is the number of data points
classes is the number of classes
        logits ([type]): [description]
    """