#!/usr/bin/env python3
""" Module that creates a confusion matrix """

def create_confusion_matrix(labels, logits):
    """Function that creates confusion matrix

    Args:
        labels ([numpy.ndarray]): array with shape (m, classes)
            containing the correct labels for each data point
            m is the number of data points
            classes is the number of classes
        logits (numpy.ndarray):  shape (m, classes) containing 
            the predicted labels
    Returns: 
        a confusion numpy.ndarray of shape (classes, classes)
        with row indices representing the correct labels and column indices representing the predicted labels        
    """
    conf_matrix