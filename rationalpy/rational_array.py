from typing import List, Tuple, Union

import numpy as np

from .rational_array_class import RationalArray


def asnumpy(
    rarr: RationalArray,
    dtype: np.dtype = None,
) -> np.ndarray:
    """
    Convert a RationalArray to a numpy array.

    Args:
        rarr (RationalArray): The RationalArray to convert.
            dtype (np.dtype, optional): The data type of the output numpy array.
        If not provided, the data type is inferred from the RationalArray.

    Returns:
        np.ndarray: A numpy array with the same shape as the RationalArray.
    """
    return np.array(rarr, dtype=dtype)


def rational_array(
    numerator: Union[int, List, Tuple, np.ndarray],
    denominator: Union[int, List, Tuple, np.ndarray] = None,
    auto_simplify: bool = True,
    dtype: np.dtype = None,
    copy: bool = True,
    order: str = "K",
    ndmin: int = 0,
) -> RationalArray:
    """
    Factory function to create a RationalArray.

    Args:
        numerator (array_like): The numerator values. Must be an integer array.
        denominator (array_like): The denominator values. If not provided,
            the denominator is set to 1. Must be an integer array.
        auto_simplify (bool): Whether to simplify the RationalArray after
            it is initialized.
        dtype (np.dtype, optional): The data type of the `numerator` and
            `denominator` arrays. If not provided, the data type is inferred from
            input arrays. Must be an integer data type. Passed to numpy.array.
        copy (bool): Whether to copy the input arrays when defining the
            `numerator` and `denominator` attributes. Passed to numpy.array.
        order (str): The memory layout of the `numerator` and
            `denominator` attributes. Passed to numpy.array.
        ndmin (int): The minimum number of dimensions of the `numerator`
            and `denominator` attributes. Passed to numpy.array.

    Returns:
        RationalArray: A RationalArray object.

    Notes:
        The `dtype`, `copy`, `order`, and `ndmin` arguments are passed to the
            numpy.array function when defining the `numerator` and `denominator`
            attributes. See the numpy.array documentation for more information.
    """
    return RationalArray(
        numerator=numerator,
        denominator=denominator,
        auto_simplify=auto_simplify,
        dtype=dtype,
        copy=copy,
        order=order,
        ndmin=ndmin,
    )
