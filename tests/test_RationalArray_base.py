import numpy as np
import pytest

from rationalpy import RationalArray


def test_RationalArray_init_int():
    ra = RationalArray(3, 4)
    assert ra.numerator == np.array(3)
    assert ra.denominator == np.array(4)


def test_RationalArray_init_tuple():
    ra = RationalArray((1, 2), (3, 4))
    assert np.array_equal(ra.numerator, np.array((1, 1)))
    assert np.array_equal(ra.denominator, np.array((3, 2)))


def test_RationalArray_init_list():
    ra = RationalArray([1, 2], [3, 4])
    assert np.array_equal(ra.numerator, np.array([1, 1]))
    assert np.array_equal(ra.denominator, np.array([3, 2]))


def test_RationalArray_init_array():
    ra = RationalArray(np.array([1, 2]), np.array([3, 4]))
    assert np.array_equal(ra.numerator, np.array([1, 1]))
    assert np.array_equal(ra.denominator, np.array([3, 2]))


def test_RationalArray_invalid_denominator_zero():
    with pytest.raises(ZeroDivisionError, match="Denominator elements cannot be 0."):
        RationalArray(np.array([1, 2]), np.array([0, 4]))


def test_RationalArray_invalid_NaN():
    with pytest.raises(
        ValueError, match="Numerator and denominator must not contain NaN."
    ):
        RationalArray(np.array([1, 2]), np.array([3, np.nan]))


def test_RationalArray_invalid_numerator_dtype():
    with pytest.raises(
        TypeError, match="Numerator and denominator must be of integer type."
    ):
        RationalArray(np.array([1.5, 2.0]), np.array([3, 4]))


def test_RationalArray_invalid_shape_mismatch():
    with pytest.raises(
        ValueError, match="Numerator and denominator must have the same shape."
    ):
        RationalArray(np.array([1, 2, 3, 4]), np.array([1, 2, 3]))


def test_RationalArray_simplify_inplace():
    ra = RationalArray(np.array([6, 8]), np.array([9, -12]))
    ra.simplify()
    print(ra)
    assert np.array_equal(ra.numerator, np.array([2, -2]))
    assert np.array_equal(ra.denominator, np.array([3, 3]))


def test_RationalArray_auto_simplify_False():
    ra = RationalArray(np.array([1, 2]), np.array([3, 4]), auto_simplify=False)
    assert np.array_equal(ra.numerator, np.array([1, 2]))
    assert np.array_equal(ra.denominator, np.array([3, 4]))


def test_RationalArray_simplify_not_inplace():
    ra = RationalArray(np.array([6, 8]), np.array([9, -12]), auto_simplify=False)
    simplified = ra.simplify(inplace=False)
    assert np.array_equal(simplified.numerator, np.array([2, -2]))
    assert np.array_equal(simplified.denominator, np.array([3, 3]))
    assert np.array_equal(ra.numerator, np.array([6, 8]))
    assert np.array_equal(ra.denominator, np.array([9, -12]))


def test_RationalArray_form_common_denominator_inplace():
    ra = RationalArray(np.array([1, 1]), np.array([3, 5]))
    ra.form_common_denominator()
    assert np.array_equal(ra.numerator, np.array([5, 3]))
    assert np.array_equal(ra.denominator, np.array([15, 15]))


def test_RationalArray_form_common_denominator_not_inplace():
    ra = RationalArray(np.array([1, 1]), np.array([3, 5]))
    result = ra.form_common_denominator(inplace=False)
    assert np.array_equal(result.numerator, np.array([5, 3]))
    assert np.array_equal(result.denominator, np.array([15, 15]))
    assert np.array_equal(ra.numerator, np.array([1, 1]))
    assert np.array_equal(ra.denominator, np.array([3, 5]))


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


def test_RationalArray_asnumpy():
    ra = RationalArray(np.array([1, 2]), np.array([3, 4]))
    result = ra.asnumpy()
    assert np.allclose(result, np.array([1 / 3, 2 / 4]))


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


def test_RationalArray_reciprocal():
    ra = RationalArray(np.array([1, 2]), np.array([3, 4]))
    result = ra.reciprocal()
    assert np.array_equal(result.numerator, np.array([3, 2]))
    assert np.array_equal(result.denominator, np.array([1, 1]))


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
