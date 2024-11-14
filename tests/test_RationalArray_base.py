from typing import List, Tuple, Union

import numpy as np
import pytest

from rationalpy import RationalArray


@pytest.mark.parametrize("numerator", [3, (1, 2), [1, 2], np.array([1, 2])])
@pytest.mark.parametrize("denominator", [None, 4, (3, 4), [3, 4], np.array([3, 4])])
@pytest.mark.parametrize("auto_simplify", [True, False])
@pytest.mark.parametrize("dtype", [None, np.int32, np.int64, np.longlong])
@pytest.mark.parametrize("copy", [True, False])
@pytest.mark.parametrize("order", ["C", "F", "A", "K"])
@pytest.mark.parametrize("ndmin", [0, 1, 2])
def test_RationalArray_init(
    numerator: Union[int, List, Tuple, np.ndarray],
    denominator: Union[int, List, Tuple, np.ndarray],
    auto_simplify: bool,
    dtype: np.dtype,
    copy: bool,
    order: str,
    ndmin: int,
):
    """
    Test RationalArray initialization with various configurations.
    """
    # Skip unsupported configurations
    if not copy and (
        not isinstance(numerator, np.ndarray) or not isinstance(denominator, np.ndarray)
    ):
        pytest.skip("Non-ndarray inputs not supported with copy=False")
    if dtype is not None and not copy:
        pytest.skip("dtype argument requires copy=True")
    if order not in ["C", "F"] and not copy:
        pytest.skip("Order other than 'C' and 'F' requires copy=True")
    if ndmin != 0 and not copy:
        pytest.skip("ndmin != 0 requires copy=True")

    # Initialize RationalArray and target values
    rarr = (
        RationalArray(
            numerator,
            denominator,
            auto_simplify=auto_simplify,
            dtype=dtype,
            copy=copy,
            order=order,
            ndmin=ndmin,
        )
        if denominator is not None
        else RationalArray(
            numerator,
            auto_simplify=auto_simplify,
            dtype=dtype,
            copy=copy,
            order=order,
            ndmin=ndmin,
        )
    )
    num_array = np.array(numerator, dtype=dtype, copy=copy, ndmin=ndmin)
    denom_array = np.array(
        denominator if denominator is not None else 1,
        dtype=dtype,
        order=order,
        ndmin=ndmin,
    )

    # Expected values for assertions
    target_shape = np.broadcast_shapes(num_array.shape, denom_array.shape)
    target_dtype = np.int64 if dtype is None else dtype

    # Assertions
    assert rarr.shape == target_shape
    assert rarr.dtype == target_dtype
    assert np.issubdtype(rarr.numerator.dtype, np.integer)
    assert np.issubdtype(rarr.denominator.dtype, np.integer)
    assert np.array_equal(np.array(rarr), num_array / denom_array)


def test_RationalArray_init_copy_False():
    """Test RationalArray initialization with copy=False."""
    numerator = np.array([1, 2])
    denominator = np.array([3, 4])
    rarr = RationalArray(numerator, denominator, copy=False)
    numerator[0] = 5
    denominator[1] = 7
    assert np.array_equal(rarr.numerator, numerator)
    assert np.array_equal(rarr.denominator, denominator)


"""Test invalid initialization configurations"""


def test_RationalArray_invalid_init_0_denominator():
    with pytest.raises(ZeroDivisionError, match="Denominator elements cannot be 0."):
        RationalArray(np.array([1, 2]), np.array([0, 4]))


def test_RationalArray_invalid_init_NaN_numerator():
    with pytest.raises(ValueError, match="Numerator elements cannot be NaN."):
        RationalArray(np.array([1, np.nan]), np.array([3, 4]))


def test_RationalArray_invalid_init_NaN_denominator():
    with pytest.raises(ValueError, match="Denominator elements cannot be NaN."):
        RationalArray(np.array([1, 2]), np.array([3, np.nan]))


def test_RationalArray_invalid_init_non_integer_numerator():
    with pytest.raises(TypeError, match="Numerator elements must be integers."):
        RationalArray(np.array([1.5, 2]), np.array([3, 4]))


def test_RationalArray_invalid_init_non_integer_denominator():
    with pytest.raises(TypeError, match="Denominator elements must be integers."):
        RationalArray(np.array([1, 2]), np.array([3.5, 4]))


def test_RationalArray_invalid_init_dtype_mismatch():
    with pytest.raises(
        ValueError, match="Numerator and denominator must have the same dtype."
    ):
        RationalArray(
            np.array([1, 2], dtype=np.int32), np.array([3, 4], dtype=np.int64)
        )


def test_RationalArray_invalid_init_shape_mismatch():
    with pytest.raises(
        ValueError, match="Unable to broadcast numerator and denominator arrays."
    ):
        RationalArray(np.array([1, 2, 3]), np.array([1, 2]))


"""
Test RationalArray methods
"""


@pytest.mark.parametrize("auto_simplify", [True, False])
@pytest.mark.parametrize("inplace", [True, False])
def test_RationalArray_simplify(auto_simplify, inplace):
    initial_numerator = np.array([6, 8])
    initial_denominator = np.array([9, -12])
    simplified_numerator = np.array([2, -2])
    simplified_denominator = np.array([3, 3])

    rarr = RationalArray(
        initial_numerator, initial_denominator, auto_simplify=auto_simplify
    )

    if auto_simplify is False:
        assert np.array_equal(rarr.numerator, initial_numerator)
        assert np.array_equal(rarr.denominator, initial_denominator)

    rarr_simplified = rarr.simplify(inplace=inplace)

    if inplace:
        assert rarr_simplified is None
        assert np.array_equal(rarr.numerator, simplified_numerator)
        assert np.array_equal(rarr.denominator, simplified_denominator)
    else:
        assert np.array_equal(rarr_simplified.numerator, simplified_numerator)
        assert np.array_equal(rarr_simplified.denominator, simplified_denominator)


@pytest.mark.parametrize("ndmin", [0, 1, 2, 3])
def test_RationalArray_printing(ndmin):
    rarr = RationalArray(np.array([1, 2]), np.array([3, 4]), ndmin=ndmin)
    try:
        repr(rarr)
    except Exception:
        pytest.fail("__repr__ failed for RationalArray object.")
    try:
        str(rarr)
    except Exception:
        pytest.fail("__str__ failed for RationalArray object.")


def test_RationalArray_getitem():
    ra = RationalArray(
        np.full((5, 5, 5), dtype=int, fill_value=1),
        np.full((5, 5, 5), dtype=int, fill_value=2),
    )
    result = ra[4:, 4:, 4:]
    assert np.all(result == RationalArray(np.array([1]), np.array([2])))


def test_RationalArray_setitem():
    ra = RationalArray(np.zeros((5, 5, 5), dtype=int), 1)
    ra[:4, :4, :4] = RationalArray(1, 64)
    assert np.all(np.sum(ra) == 1)


def test_RationalArray_setitem_with_tuple():
    ra = RationalArray(np.zeros((5, 5, 5), dtype=int), 1)
    ra[:4, :4, :4] = (1, 64)
    assert np.all(np.sum(ra) == 1)


def test_RationalArray_setitem_with_scalar():
    ra = RationalArray(np.zeros((5, 5, 5), dtype=int), 1)
    ra[:4, :4, :4] = 1
    assert np.all(np.sum(ra) == 64)


def test_RationalArray_invalid_setitem():
    ra = RationalArray(np.zeros((5, 5, 5), dtype=int), 1)
    value = "invalid"
    with pytest.raises(
        ValueError, match=f"Cannot assign from type {type(value)} to RationalArray."
    ):
        ra[:4, :4, :4] = value


@pytest.mark.parametrize("inplace", [True, False])
def test_RationalArray_form_common_denominator(inplace):
    initial_numerator = np.array([1, 1])
    initial_denominator = np.array([3, 5])
    common_numerator = np.array([5, 3])
    common_denominator = np.array([15, 15])

    rarr = RationalArray(initial_numerator, initial_denominator)
    common_rarr = rarr.form_common_denominator(inplace=inplace)

    if inplace:
        assert common_rarr is None
        assert np.array_equal(rarr.numerator, common_numerator)
        assert np.array_equal(rarr.denominator, common_denominator)
    else:
        assert np.array_equal(rarr.numerator, initial_numerator)
        assert np.array_equal(rarr.denominator, initial_denominator)
        assert np.array_equal(common_rarr.numerator, common_numerator)
        assert np.array_equal(common_rarr.denominator, common_denominator)


def test_RationalArray_reciprocal():
    ra = RationalArray(np.array([1, 2]), np.array([3, 4]))
    result = ra.reciprocal()
    assert np.array_equal(result.numerator, np.array([3, 2]))
    assert np.array_equal(result.denominator, np.array([1, 1]))


def test_RationalArray_decompose():
    ra = RationalArray(np.array([1, 1]), np.array([3, 5]))
    result = ra.decompose()
    assert np.all(result[0] == RationalArray(np.array([1, 1])))
    assert np.all(result[1] == RationalArray(1, np.array([3, 5])))


def test_RationalArray_asratio():
    ra = RationalArray(np.array([1, 1]), np.array([3, 5]))
    numerator, denominator = ra.asratio()
    assert np.array_equal(numerator, np.array([1, 1]))
    assert np.array_equal(denominator, np.array([3, 5]))


"""Test RationalArray arithmetic operations"""


def test_RationalArray_add():
    """Test addition of two RationalArray objects.
    1/3 + 2/4 = 5/6
    2/4 + 3/5 = 11/10
    """
    ra1 = RationalArray(np.array([1, 2]), np.array([3, 4]))
    ra2 = RationalArray(np.array([2, 3]), np.array([4, 5]))
    result = ra1 + ra2
    assert np.array_equal(result.numerator, np.array([5, 11]))
    assert np.array_equal(result.denominator, np.array([6, 10]))


def test_RationalArray_add_with_int_scalar():
    """Test addition of RationalArray with integer scalar.
    1/3 + 2 = 7/3
    2/4 + 2 = 3/2
    """
    ra = RationalArray(np.array([1, 2]), np.array([3, 4]))
    result = ra + 2
    assert np.array_equal(result.numerator, np.array([7, 5]))
    assert np.array_equal(result.denominator, np.array([3, 2]))


def test_RationalArray_sub():
    """Test subtraction of two RationalArray objects.
    1/3 - 2/4 = -1/6
    2/4 - 3/5 = -1/10
    """
    ra1 = RationalArray(np.array([1, 2]), np.array([3, 4]))
    ra2 = RationalArray(np.array([2, 3]), np.array([4, 5]))
    result = ra1 - ra2
    assert np.array_equal(result.numerator, np.array([-1, -1]))
    assert np.array_equal(result.denominator, np.array([6, 10]))


def test_RationalArray_mul():
    """Test multiplication of two RationalArray objects.
    1/3 * 2/4 = 1/6
    2/4 * 3/5 = 3/10
    """
    ra1 = RationalArray(np.array([1, 2]), np.array([3, 4]))
    ra2 = RationalArray(np.array([2, 3]), np.array([4, 5]))
    result = ra1 * ra2
    assert np.array_equal(result.numerator, np.array([1, 3]))
    assert np.array_equal(result.denominator, np.array([6, 10]))


def test_RationalArray_mul_with_int_numpy_array():
    """Test multiplication of RationalArray with numpy array of integers.
    1/3 * 1 = 1/3
    2/4 * 2 = 1
    """
    ra = RationalArray(np.array([1, 2]), np.array([3, 4]))
    result = ra * np.array([1, 2])
    assert np.array_equal(result.numerator, np.array([1, 1]))
    assert np.array_equal(result.denominator, np.array([3, 1]))


def test_RationalArray_mul_with_float_numpy_array():
    """Test multiplication of RationalArray with numpy array of floats.
    1/3 * 1.0 = 0.3333...
    2/4 * 2.0 = 1.0
    """
    ra = RationalArray(np.array([1, 2]), np.array([3, 4]))
    result = ra * np.array([1.0, 2.0])
    assert np.array_equal(result, np.array([1 / 3, 1.0]))


def test_RationalArray_mul_with_int_scalar():
    """Test multiplication of RationalArray with integer scalar.
    1/3 * 2 = 2/3
    2/4 * 2 = 1
    """
    ra = RationalArray(np.array([1, 2]), np.array([3, 4]))
    result = ra * 2
    print(result)
    assert np.array_equal(result.numerator, np.array([2, 1]))
    assert np.array_equal(result.denominator, np.array([3, 1]))


def test_RationalArray_mul_with_float_scalar():
    """Test multiplication of RationalArray with float scalar.
    1/3 * 2.0 = 0.6666...
    2/4 * 2.0 = 1.0
    """
    ra = RationalArray(np.array([1, 2]), np.array([3, 4]))
    result = ra * 2.0
    assert np.array_equal(result, np.array([2 / 3, 1.0]))


@pytest.mark.parametrize(
    "arg2",
    [RationalArray(np.array([2, 3]), np.array([4, 5])), np.array([1.0, 2.0]), 2, 2.0],
)
def test_RationalArray_mul_commutativity(arg2):
    arg1 = RationalArray(np.array([1, 2]), np.array([3, 4]))
    result1 = arg1 * arg2
    result2 = arg2 * arg1
    assert np.all(result1 == result2)


def test_RationalArray_div():
    """Test division of two RationalArray objects.
    1/3 / 2/4 = 2/3
    2/4 / 3/5 = 5/6
    """
    ra1 = RationalArray(np.array([1, 2]), np.array([3, 4]))
    ra2 = RationalArray(np.array([2, 3]), np.array([4, 5]))
    result = ra1 / ra2
    assert np.array_equal(result.numerator, np.array([2, 5]))
    assert np.array_equal(result.denominator, np.array([3, 6]))


def test_RationalArray_div_with_int_numpy_array():
    """Test division of RationalArray with numpy array of integers.
    1/3 / 1 = 1/3
    2/4 / 2 = 1/4
    """
    ra = RationalArray(np.array([1, 2]), np.array([3, 4]))
    result = ra / np.array([1, 2])
    assert np.array_equal(result.numerator, np.array([1, 1]))
    assert np.array_equal(result.denominator, np.array([3, 4]))


def test_RationalArray_div_with_float_numpy_array():
    """Test division of RationalArray with numpy array of floats.
    1/3 / 1.0 = 0.3333...
    2/4 / 2.0 = 0.25
    """
    ra = RationalArray(np.array([1, 2]), np.array([3, 4]))
    result = ra / np.array([1.0, 2.0])
    assert np.array_equal(result, np.array([1 / 3, 1 / 4]))


def test_RationalArray_div_with_int_scalar():
    """Test division of RationalArray with integer scalar.
    1/3 / 2 = 1/6
    2/4 / 2 = 1/4
    """
    ra = RationalArray(np.array([1, 2]), np.array([3, 4]))
    result = ra / 2
    assert np.array_equal(result.numerator, np.array([1, 1]))
    assert np.array_equal(result.denominator, np.array([6, 4]))


def test_RationalArray_div_with_float_scalar():
    """Test division of RationalArray with float scalar.
    1/3 / 2.0 = 0.1666...
    2/4 / 2.0 = 0.25
    """
    ra = RationalArray(np.array([1, 2]), np.array([3, 4]))
    result = ra / 2.0
    assert np.array_equal(result, np.array([1 / 6, 1 / 4]))


def test_RationalArray_negate():
    ra = RationalArray(np.array([1, 2]), np.array([3, 4]))
    result = -ra
    assert np.array_equal(result.numerator, np.array([-1, -1]))
    assert np.array_equal(result.denominator, np.array([3, 2]))


"""Test conversion to numpy array"""


@pytest.mark.parametrize("dtype", [None, np.int32, np.int64, np.longlong])
def test_RationalArray_array(dtype):
    ra = RationalArray(np.array([1, 2]), np.array([3, 4]))
    assert np.allclose(np.array(ra, dtype=dtype), np.array([1 / 3, 2 / 4], dtype=dtype))
