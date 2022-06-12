"""Module containing the anomaly threshold function."""
from typing import List
import numpy as np


def anomaly_threshold(confidence_list: List[float]) -> float:
    """Obtain the list index of top and bottom threshold of a band.

    :param confidence_list: The time-series confidence from
                            the energy based model.
    :return:                The bottom threshold value.
    """
    confidence_np = np.array(confidence_list)
    list_mean = np.mean(confidence_np)
    list_std = np.std(confidence_np)
    bottom_threshold = list_mean - 3 * list_std
    return bottom_threshold
