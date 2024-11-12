import numpy as np

from rationalpy import RationalArray


def test_RationalArray_numpy_abs():
    """Test numpy.abs implementation for RationalArray."""
    ra = RationalArray(np.array([1, -2]), np.array([3, 4]))
    result = np.abs(ra)
    assert np.array_equal(result.numerator, np.array([1, 1]))
    assert np.array_equal(result.denominator, np.array([3, 2]))


def test_RationalArray_numpy_append():
    """Test numpy.append implementation for RationalArray."""
    ra1 = RationalArray(np.array([1, 2]), np.array([3, 4]), auto_simplify=False)
    ra2 = RationalArray(np.array([2, 3]), np.array([4, 5]), auto_simplify=False)
    result = np.append(ra1, ra2)
    assert np.array_equal(result.numerator, np.array([1, 2, 2, 3]))
    assert np.array_equal(result.denominator, np.array([3, 4, 4, 5]))


def test_RationalArray_numpy_concatenate():
    """Test numpy.concatenate implementation for RationalArray."""
    ra1 = RationalArray(np.array([1, 2]), np.array([3, 4]), auto_simplify=False)
    ra2 = RationalArray(np.array([2, 3]), np.array([4, 5]), auto_simplify=False)
    result = np.concatenate([ra1, ra2])
    assert np.array_equal(result.numerator, np.array([1, 2, 2, 3]))
    assert np.array_equal(result.denominator, np.array([3, 4, 4, 5]))


def test_RationalArray_numpy_full_like():
    """Test numpy.full_like implementation for RationalArray."""
    ra = RationalArray(np.array([1, 2]), np.array([3, 4]))
    result = np.full_like(ra, 5)
    assert np.array_equal(result.numerator, np.array([5, 5]))
    assert np.array_equal(result.denominator, np.array([1, 1]))


def test_RationalArray_numpy_insert():
    """Test numpy.insert implementation for RationalArray."""
    ra1 = RationalArray(np.array([1, 2]), np.array([3, 4]), auto_simplify=False)
    ra2 = RationalArray(np.array([2]), np.array([4]), auto_simplify=False)
    result = np.insert(ra1, 1, ra2)
    assert np.array_equal(result.numerator, np.array([1, 2, 2]))
    assert np.array_equal(result.denominator, np.array([3, 4, 4]))


def test_RationalArray_numpy_mean():
    """Test numpy.mean implementation for RationalArray."""
    ra = RationalArray(np.array([1, 2]), np.array([3, 4]))
    result = np.mean(ra)
    assert np.array_equal(result.numerator, np.array(5))
    assert np.array_equal(result.denominator, np.array(12))


def test_RationalArray_numpy_nonzero():
    """Test numpy.nonzero implementation for RationalArray."""
    ra = RationalArray(np.array([1, 0]), np.array([3, 4]))
    result = np.nonzero(ra)
    assert np.array_equal(result, (np.array([0]),))


def test_RationalArray_numpy_square():
    """Test numpy.square implementation for RationalArray."""
    ra = RationalArray(np.array([5, 6]), np.array([5, 7]))
    result = np.square(ra)
    assert np.array_equal(result.numerator, np.array([1, 36]))
    assert np.array_equal(result.denominator, np.array([1, 49]))


def test_RationalArray_numpy_sum():
    """Test numpy.sum implementation for RationalArray."""
    ra = RationalArray(np.array([1, 2]), np.array([3, 4]))
    result = np.sum(ra)
    assert np.array_equal(result.numerator, np.array(5))
    assert np.array_equal(result.denominator, np.array(6))
