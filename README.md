# RationalPy

`rationalpy` is a Python library and extension of numpy designed for creating and manipulating arrays of rational numbers, represented as exact fractions. It enables users to perform arithmetic operations on arrays while maintaining an exact rational representation.

## Table of Contents
- [RationalPy](#rationalpy)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Features](#features)
  - [Examples](#examples)
  - [License](#license)

---

## Installation

To install `rationalpy`, clone the repository and use `pip`:

```bash
git clone https://github.com/jpalafou/rationalpy.git
cd rationalpy
pip install .
```

For optional development dependencies (e.g., testing and documentation), install with:

```pip install .[test,docs]```

## Usage

```py
import rationalpy as rp

# Create a 1D RationalArray object
rarr = rp.RationalArray([1, 2], [2, 3])
print(rarr)
# Output: [1/2 2/3]

# Create a 2D array
rarr_2d = rp.RationalArray([[1, 2], [3, 4]], [[4, 5], [6, 7]])
print(rarr_2d)
# Output: [[1/4 2/5]
#          [1/2 4/7]]
```

Note that the arrays are automatically simplified upon instantiation. This behavior can be disabled by setting `auto_simplify=False`.

## Features

* Supports exact arithmetic with fractions.
* Provides numpy-style broadcasting for arrays of rational numbers.
* Supports common arithmetic operations (addition, subtraction, multiplication, and division) as well as some numpy array functions (`np.sum`, `np.concatenate`, and more).
* Includes methods for fraction simplification `RationalArray.simplify()` and finding the least common denominator `RationalArray.form_common_denominator()`.
* Customizable `__repr__` for displaying rational arrays in a clean, aligned format.

## Examples

Simple arithmetic:

```py
import rationalpy as rp

rarr1 = rp.RationalArray([0, 1, 2, 3, 4], [1, 2, 3, 4, 5])
print(rarr1)
# Output: [0/1 1/2 2/3 3/4 4/5]

rarr2 = rp.RationalArray(1, [1, 2, 3, 4, 5])
print(rarr2)
# Output: [1/1 1/2 1/3 1/4 1/5]

print(rarr1 + rarr2)
# Output: [1/1 1/1 1/1 1/1 1/1]

print(rarr1 - rarr2)
# Output: [-1/1 0/1 1/3 1/2 3/5]

print(rarr1 * rarr2)
# Output: [0/1 1/4 2/9 3/16 4/25]

print(rarr1 / rarr2)
# Output: [0/1 1/1 2/1 3/1 4/1]
```

## License

This project is licensed under the MIT License. See [LICENSE]() for more information.
