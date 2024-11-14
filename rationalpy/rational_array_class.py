"""
This module provides functionality for handling arrays of rational numbers.

It includes a class `RationalArray` that supports various arithmetic operations
with rational numbers, numpy arrays, integers, and floats. The module also
contains helper functions and a registry for array functions.

Classes:
    RationalArray: A class for performing arithmetic operations on arrays of rational
        numbers.

Constants:
    HANDLED_FUNCTIONS: A registry for array functions.
"""

from typing import List, Optional, Tuple, Union, cast

import numpy as np

# array function registry
HANDLED_FUNCTIONS = {}


def implements(np_function):
    """Decorator to register a function implementation for a NumPy ufunc. Functions are
    registered to the HANDLED_FUNCTIONS dictionary.
    """

    def decorator(func):
        HANDLED_FUNCTIONS[np_function] = func
        return func

    return decorator


class RationalArray(np.lib.mixins.NDArrayOperatorsMixin):
    """A class to represent arrays of rational numbers."""

    def __init__(
        self,
        numerator: Union[np.ndarray, list, tuple, int],
        denominator: Union[np.ndarray, list, tuple, int] = 1,
        auto_simplify: bool = True,
        dtype: Optional[np.dtype] = None,
        copy: bool = True,
        order: str = "K",
        ndmin: int = 0,
    ):
        """Initialize RationalArray with a numerator and denominator.

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

        Notes:
            The `dtype`, `copy`, `order`, and `ndmin` arguments are passed to the
                numpy.array function when defining the `numerator` and `denominator`
                attributes. See the numpy.array documentation for more information.
        """
        # Avoid copy=False for non-ndarray inputs
        if copy is False and not (
            isinstance(numerator, np.ndarray) and isinstance(denominator, np.ndarray)
        ):
            raise ValueError("Cannot set copy=False for non-ndarray inputs.")

        # Declare numerator and denominator attributes as numpy arrays
        self.numerator = np.array(
            numerator, dtype=dtype, copy=copy, order=order, ndmin=ndmin
        )
        self.denominator = np.array(
            denominator, dtype=dtype, copy=copy, order=order, ndmin=ndmin
        )

        # Handle scalar inputs
        if self.numerator.size == 1 and self.denominator.size > 1:
            self.numerator = np.full_like(self.denominator, self.numerator.flat[0])
        if self.denominator.size == 1 and self.numerator.size > 1:
            self.denominator = np.full_like(self.numerator, self.denominator.flat[0])

        # Broadcast numerator and denominator arrays
        try:
            self.numerator, self.denominator = np.broadcast_arrays(
                self.numerator, self.denominator
            )
        except ValueError:
            raise ValueError("Unable to broadcast numerator and denominator arrays.")
        self.numerator.setflags(write=True)
        self.denominator.setflags(write=True)

        # Initialize attributes
        self.auto_simplify = auto_simplify
        self.validate(simplify=auto_simplify)

    def validate(self, simplify: Optional[bool] = None):
        """
        Validate the RationalArray arrays with optional simplification.

        Args:
            simplify (bool, optional): Whether to simplify the RationalArray after
                validation. If not provided, the auto_simplify attribute is used.
        """
        # Check for invalid values
        if np.any(self.denominator == 0):
            raise ZeroDivisionError("Denominator elements cannot be 0.")
        if np.any(np.isnan(self.numerator)):
            raise ValueError("Numerator elements cannot be NaN.")
        if np.any(np.isnan(self.denominator)):
            raise ValueError("Denominator elements cannot be NaN.")
        if not np.issubdtype(self.numerator.dtype, np.integer):
            raise TypeError("Numerator elements must be integers.")
        if not np.issubdtype(self.denominator.dtype, np.integer):
            raise TypeError("Denominator elements must be integers.")

        # Assign attributes if input arrays are valid
        if self.numerator.dtype != self.denominator.dtype:
            raise ValueError("Numerator and denominator must have the same dtype.")
        else:
            self.dtype = self.numerator.dtype
        if self.numerator.shape != self.denominator.shape:
            raise ValueError("Numerator and denominator must have the same shape.")
        else:
            self.ndim = self.numerator.ndim
            self.shape = self.numerator.shape
            self.size = self.numerator.size

        # Simplify the RationalArray if auto_simplify is True
        simplify = self.auto_simplify if simplify is None else simplify
        if simplify:
            self.simplify()

    def simplify(self, inplace: bool = True):
        """Element-wise rational array simplification.

        Args:
            inplace (bool, optional): Whether to perform the operation in-place.
        """
        rarr = self if inplace else self.copy()

        # First, simplify negatives
        rarr._simplify_negatives()

        # Then, simplify each fraction
        gcd = np.gcd(rarr.numerator, rarr.denominator)
        rarr.numerator //= gcd
        rarr.denominator //= gcd

        if not inplace:
            return rarr

    def _simplify_negatives(self, inplace: bool = True):
        """Move negative signs from the denominator to the numerator.

        Args:
            inplace (bool, optional): Whether to perform the operation in-place.
        """
        rarr = self if inplace else self.copy()
        if not np.any(rarr.denominator < 0):
            return rarr if not inplace else None
        positivity_factor = np.where(rarr.denominator < 0, -1, 1)
        rarr.numerator *= positivity_factor
        rarr.denominator = np.abs(rarr.denominator)
        if not inplace:
            return rarr

    def __repr__(
        self,
    ) -> str:
        """
        Return a string representation of the RationalArray containing the numerator,
            denominator, and auto_simplify attribute.
        """
        return self._construct_string_array(
            prefix="RationalArray(",
            suffix=f", auto_simplify={self.auto_simplify})",
            separator=", ",
        )

    def __str__(self) -> str:
        """
        Return a string representation of the RationalArray containing the numerator
            and denominator.
        """
        return self._construct_string_array(separator=" ")

    def _construct_string_array(
        self, prefix: str = "", suffix: str = "", **kwargs
    ) -> str:
        """
        Construct a generix string representation of the RationalArray.

        Args:
            prefix (str, optional): Prefix for the string representation. Passed to
                numpy.array2string.
            suffix (str, optional): Suffix for the string representation. Passed to
                numpy.array2string.
            **kwargs: Additional keyword arguments passed to numpy.array2string.
        """
        idxs = np.arange(self.size).reshape(self.shape)
        nums = self.numerator.flatten()
        dens = self.denominator.flatten()
        return (
            prefix
            + np.array2string(
                idxs,
                formatter={"int": lambda i: f"{nums[i]}/{dens[i]}"},
                prefix=prefix,
                suffix=suffix,
                **kwargs,
            )
            + suffix
        )

    def __getitem__(self, key):
        """
        Get elements of the RationalArray.

        Args:
            key: The key to index the RationalArray.
        """
        return self.__class__(
            self.numerator[key], self.denominator[key], auto_simplify=False
        )

    def __setitem__(self, key, value: Union["RationalArray", Tuple[int, int], int]):
        """
        Set elements of the RationalArray.

        Args:
            key: The key to index the RationalArray.
            value (Union[RationalArray, Tuple[int, int], int])): The value to assign
                to the RationalArray.
                - RationalArray: Assign from another RationalArray.
                - Tuple[int, int]: Assign scalar rational from a tuple of integers
                    (numerator, denominator).
                - int: Assign scalar rational where the denominator is 1.
        """
        if type(value) is self.__class__:
            # Assign from another RationalArray
            assigned_numerator = value.numerator
            assigned_denominator = value.denominator
        elif isinstance(value, tuple) or isinstance(value, list):
            if len(value) != 2:
                raise ValueError(
                    "Tuple must have two elements: (numerator, denominator"
                )
            assigned_numerator, assigned_denominator = value
        elif np.issubdtype(type(value), np.integer):
            assigned_numerator, assigned_denominator = value, 1
        else:
            raise ValueError(f"Cannot assign from type {type(value)} to RationalArray.")

        self.numerator[key] = assigned_numerator
        self.denominator[key] = assigned_denominator

    def form_common_denominator(self, inplace: bool = True):
        """Form a common denominator for the rational array.

        Args:
            inplace (bool, optional): Whether to perform the operation in-place.
        """
        rarr = self if inplace else self.copy()

        # Simplify negatives
        rarr._simplify_negatives()

        # Find common denominator
        lcm = np.lcm.reduce(rarr.denominator)
        rarr.numerator *= lcm // rarr.denominator
        rarr.denominator = np.full_like(rarr.denominator, lcm)

        if not inplace:
            return rarr

    def reciprocal(self) -> "RationalArray":
        """
        Return the reciprocal of the RationalArray (1 / self).
        """
        return self.__class__(self.denominator, self.numerator)

    def decompose(self) -> Tuple["RationalArray", "RationalArray"]:
        """
        Decompose the rational array into two parts: n_i / 1 and 1 / d_i.

        Returns:
            Tuple[RationalArray, RationalArray]: The decomposed RationalArrays.
        """
        return self.__class__(self.numerator, 1), self.__class__(1, self.denominator)

    def asratio(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the RationalArray as a tuple of numpy arrays.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The RationalArray as a tuple of numpy arrays.
        """
        return self.numerator, self.denominator

    def copy(self) -> "RationalArray":
        """Return a copy of the RationalArray."""
        return self.__class__(
            self.numerator,
            self.denominator,
            auto_simplify=self.auto_simplify,
            dtype=self.dtype,
            copy=True,
        )

    def __neg__(self) -> "RationalArray":
        return self.__class__(-self.numerator, self.denominator)

    def __add__(self, other: Union["RationalArray", int]) -> "RationalArray":
        if isinstance(other, self.__class__):
            return self._add_RationalArray(other)
        elif np.issubdtype(type(other), np.integer):
            return self._add_scalar(cast(int, other))
        return NotImplemented

    def _add_RationalArray(self, other: "RationalArray") -> "RationalArray":
        n1, d1 = self.numerator, self.denominator
        n2, d2 = other.numerator, other.denominator
        lcm = np.lcm(d1, d2)
        rarr = self.__class__(
            n1 * (lcm // d1) + n2 * (lcm // d2),
            lcm,
            auto_simplify=self.auto_simplify or other.auto_simplify,
        )
        return rarr

    def _add_scalar(self, other: int) -> "RationalArray":
        return self.__add__(self.__class__(other, 1))

    def __sub__(self, other: Union[int, "RationalArray"]) -> "RationalArray":
        return self.__add__(-other)

    def __mul__(
        self,
        other: Union["RationalArray", np.ndarray, int, float],
    ) -> Union["RationalArray", np.ndarray]:
        if isinstance(other, self.__class__):
            return self._mul_by_RationalArray(other)
        elif isinstance(other, np.ndarray):
            return self._mul_by_nparray(other)
        elif np.issubdtype(type(other), np.integer):
            return self._mul_by_int(cast(int, other))
        elif np.issubdtype(type(other), np.floating):
            return self._mul_by_float(cast(float, other))
        return NotImplemented

    def _mul_by_RationalArray(
        self,
        other: "RationalArray",
    ) -> "RationalArray":
        n1, d1 = self.numerator, self.denominator
        n2, d2 = other.numerator, other.denominator
        rarr = self.__class__(
            n1 * n2, d1 * d2, auto_simplify=self.auto_simplify or other.auto_simplify
        )
        return rarr

    def _mul_by_nparray(self, other: np.ndarray) -> Union["RationalArray", np.ndarray]:
        if np.issubdtype(other.dtype, np.integer):
            return self.__class__(self.numerator * other, self.denominator)
        return np.array(self) * other

    def _mul_by_int(self, other: int) -> "RationalArray":
        return self.__class__(self.numerator * other, self.denominator)

    def _mul_by_float(self, other: float) -> np.ndarray:
        return np.array(self) * other

    def __rmul__(
        self, other: Union["RationalArray", np.ndarray, int, float]
    ) -> Union["RationalArray", np.ndarray]:
        return self.__mul__(other)

    def __floordiv__(
        self, other: Union["RationalArray", np.ndarray, int, float]
    ) -> Union[np.ndarray, "RationalArray"]:
        if isinstance(other, self.__class__):
            return self._floordiv_by_RationalArray(other)
        elif isinstance(other, np.ndarray):
            return self._floordiv_by_nparray(other)
        elif np.issubdtype(type(other), np.integer):
            return self._floordiv_by_int(cast(int, other))
        elif np.issubdtype(type(other), np.floating):
            return self._floordiv_by_float(cast(float, other))
        return NotImplemented

    def _floordiv_by_RationalArray(self, other: "RationalArray") -> "RationalArray":
        return self * other.reciprocal()

    def _floordiv_by_nparray(
        self, other: np.ndarray
    ) -> Union[np.ndarray, "RationalArray"]:
        if np.issubdtype(other.dtype, np.integer):
            if np.any(other == 0):
                raise ZeroDivisionError("Division by zero.")
            return self * self.__class__(1, other)
        return np.array(self) / other

    def _floordiv_by_int(self, other: int) -> "RationalArray":
        return self * self.__class__(1, other)

    def _floordiv_by_float(self, other: float) -> np.ndarray:
        return np.array(self) / other

    def __truediv__(
        self, other: Union[int, float, np.ndarray, "RationalArray"]
    ) -> Union["RationalArray", np.ndarray]:
        return self.__floordiv__(other)

    def __array__(self, dtype: Optional[np.dtype] = None, copy: Optional[bool] = None):
        """
        Return the RationalArray as a numpy array.

        Args:
            dtype (np.dtype, optional): The data type of the output numpy array. If not
                provided, the data type is inferred from the RationalArray.
            copy (bool, optional): Unused parameter. Kept for compatibility with numpy.
        """
        if copy is False:
            raise ValueError("Cannot set copy=False for RationalArray.__array__.")
        return np.array(self.numerator / self.denominator, dtype=dtype)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method == "__call__":
            if ufunc not in HANDLED_FUNCTIONS:
                # fall back to NumPy ufunc
                np_inputs = [
                    np.array(x) if isinstance(x, self.__class__) else x for x in inputs
                ]
                return ufunc(*np_inputs, **kwargs)
            return HANDLED_FUNCTIONS[ufunc](*inputs, **kwargs)
        return NotImplemented

    def __array_function__(self, func, types, args, kwargs):
        if func not in HANDLED_FUNCTIONS:
            return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)


# register array function implementations
@implements(np.abs)
def abs_numpy(arr: RationalArray) -> RationalArray:
    """
    numpy.abs implementation for RationalArray.
    """
    return arr.__class__(
        np.abs(arr.numerator), np.abs(arr.denominator), arr.auto_simplify
    )


@implements(np.append)
def append(
    arr: RationalArray, values: RationalArray, axis: Optional[int] = None
) -> RationalArray:
    """numpy.append implementation for RationalArray.

    Args:
        arr (RationalArray): The input array.
        values (RationalArray): The values to append.
        axis (int, optional): The axis along which to append the values.

    """
    if not isinstance(values, RationalArray) or not isinstance(arr, RationalArray):
        raise ValueError("inputs must be RationalArrays.")
    appended_numerator = np.append(arr.numerator, values.numerator, axis=axis)
    appended_denominator = np.append(arr.denominator, values.denominator, axis=axis)
    return RationalArray(appended_numerator, appended_denominator, arr.auto_simplify)


@implements(np.concatenate)
def concatenate(arrs: List[RationalArray], axis: Optional[int] = None) -> RationalArray:
    """numpy.concatenate implementation for RationalArray.

    Args:
        arrs (List[RationalArray]): The input arrays.
        axis (int, optional): The axis along which to concatenate the arrays.

    Returns:
        RationalArray: The concatenated array.
    """
    # Collect all numerators and denominators from input arrays
    nums = [arr.numerator for arr in arrs]
    dens = [arr.denominator for arr in arrs]
    auto_simplify = any(arr.auto_simplify for arr in arrs)

    # Concatenate the numerators and denominators along the specified axis
    concatenated_numerator = np.concatenate(nums, axis=axis)
    concatenated_denominator = np.concatenate(dens, axis=axis)

    # Return a new RationalArray with the concatenated values
    return RationalArray(
        concatenated_numerator, concatenated_denominator, auto_simplify
    )


@implements(np.full_like)
def full_like(a, fill_value):
    """
    numpy.full_like implementation for RationalArray.
    """
    if not (
        isinstance(fill_value, RationalArray)
        or np.issubdtype(type(fill_value), np.integer)
    ):
        raise ValueError("fill_value must be a RationalArray or integer.")
    return RationalArray(np.ones_like(a.numerator), 1, a.auto_simplify) * fill_value


@implements(np.insert)
def insert(arr, obj, values, axis: Optional[int] = None):
    """
    numpy.insert implementation for RationalArray.
    """
    if not isinstance(values, RationalArray) or not isinstance(arr, RationalArray):
        raise ValueError("inputs must be RationalArrays.")
    res_n = np.insert(arr.numerator, obj, values.numerator)
    res_d = np.insert(arr.denominator, obj, values.denominator)
    return RationalArray(res_n, res_d, arr.auto_simplify)


@implements(np.mean)
def mean(arr):
    """
    numpy.mean implementation for RationalArray.
    """
    res = np.sum(arr) * RationalArray(1, arr.size, arr.auto_simplify)
    return res


@implements(np.multiply)
def multiply(arr1, arr2):
    """
    numpy.multiply implementation for RationalArray.
    """
    if isinstance(arr1, np.ndarray):
        return arr2 * arr1
    return arr1 * arr2


@implements(np.nonzero)
def nonzero(arr):
    """
    numpy.nonzero implementation for RationalArray.
    """
    return np.nonzero(arr.numerator)


@implements(np.square)
def square(arr):
    """
    numpy.square implementation for RationalArray.
    """
    res = RationalArray(
        np.square(arr.numerator), np.square(arr.denominator), arr.auto_simplify
    )
    return res


@implements(np.sum)
def sum_numpy(arr):
    """
    numpy.sum implementation for RationalArray.
    """
    res = arr.form_common_denominator(inplace=False)
    res = RationalArray(
        np.sum(res.numerator), res.denominator.flat[0], arr.auto_simplify
    )
    return res
