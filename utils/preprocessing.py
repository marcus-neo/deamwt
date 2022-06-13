"""Module to perform preprocessing on the data."""
from typing import Union
import pywt
import numpy as np


def wrcoef(signal, coef_type, coeffs, wavename, level):
    """Reconstructs a signal from its wavelet coefficients.

    :param signal: Input signal.
    :param coef_type: Type of the coefficients.
    :param coeffs: Wavelet coefficients.
    :param wavename: Wavelet name.
    :param level: Decomposition level.
    :return: Reconstructed signal.
    """
    signal_size = np.array(signal).size
    approx, details = coeffs[0], list(reversed(coeffs[1:]))
    if coef_type == "a":
        return pywt.upcoef("a", approx, wavename, level=level)[:signal_size]
    if coef_type == "d":
        return pywt.upcoef("d", details[level - 1], wavename, level=level)[
            :signal_size
        ]
    raise ValueError(f"Invalid coefficient type: {coef_type}")


def db1_multires_analysis(data: Union[np.ndarray, list], level: int):
    """Execute Multi-Resolution Analysis on the given data.

    Currently only supports db1 wavelet family.
    :param data: Input Array.
    :param level: Decomposition level.
    :return: Decomposed data.
    """
    
    def single_variate_decomposition(single_variate_data):
        """Perform decomposition on a single feature."""
        coeffs = pywt.wavedec(single_variate_data, "db1", level=level)
        output = wrcoef(single_variate_data, "a", coeffs, "db1", level)
        for detail_level in range(1, level + 1):
            output = np.c_[wrcoef(single_variate_data, "d", coeffs, "db1", detail_level), output]
        return output

    
    # Check number of features
    data = np.array(data)
    if len(data.shape) == 1:
        return single_variate_decomposition(data)
    elif len(data.shape) == 2:
        for ind in range(data.shape[1]):
            output = single_variate_decomposition(data[:, ind].flatten())
            if ind == 0:
                output_array = output
            else:
                output_array = np.c_[output_array, output]
        return output_array


if __name__ == "__main__":
    data = np.random.randint(0, 10, size=100)
    data2 = np.random.randint(0, 10, size=100)
    data = np.c_[data, data2]
    print(db1_multires_analysis(data, 2))
