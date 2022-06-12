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
    coeffs = pywt.wavedec(data, "db1", level=level)
    output = wrcoef(data, "a", coeffs, "db1", level)
    for detail_level in range(1, level + 1):
        output = np.c_[wrcoef(data, "d", coeffs, "db1", detail_level), output]
    return output


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("error")

    try:
        y = np.array(list(range(4)))
        print(db1_multires_analysis(y, 2))
    except UserWarning:
        print("UserWarning")
