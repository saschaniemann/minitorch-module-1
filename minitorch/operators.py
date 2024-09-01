"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


def mul(a: float, b: float) -> float:
    """Multiply two floating-point numbers.

    Args:
    ----
        a (float): The first number.
        b (float): The second number.

    Returns:
    -------
        float: The product of `a` and `b`.

    """
    return a * b


def id(a: float) -> float:
    """Return the input value unchanged.

    Args:
    ----
        a (float): The input value.

    Returns:
    -------
        float: The same value as `a`.

    """
    return a


def add(a: float, b: float) -> float:
    """Add two floating-point numbers.

    Args:
    ----
        a (float): The first number.
        b (float): The second number.

    Returns:
    -------
        float: The sum of `a` and `b`.

    """
    return a + b


def neg(a: float) -> float:
    """Return the negation of a floating-point number.

    Args:
    ----
        a (float): The number to negate.

    Returns:
    -------
        float: The negation of `a`.

    """
    return -a


def lt(a: float, b: float) -> bool:
    """Return True if the first number is less than the second.

    Args:
    ----
        a (float): The first number.
        b (float): The second number.

    Returns:
    -------
        bool: True if `a` is less than `b`, otherwise False.

    """
    return a < b


def eq(a: float, b: float) -> bool:
    """Return True if two numbers are equal.

    Args:
    ----
        a (float): The first number.
        b (float): The second number.

    Returns:
    -------
        bool: True if `a` is equal to `b`, otherwise False.

    """
    return a == b


def max(a: float, b: float) -> float:
    """Return the maximum of two floating-point numbers.

    Args:
    ----
        a (float): The first number.
        b (float): The second number.

    Returns:
    -------
        float: The larger of `a` and `b`.

    """
    return a if a > b else b


def is_close(a: float, b: float) -> bool:
    """Return True if two floating-point numbers are close to each other.

    Args:
    ----
        a (float): The first number.
        b (float): The second number.

    Returns:
    -------
        bool: True if the absolute difference between `a` and `b` is less
        than 1e-2, otherwise False.

    """
    return abs(a - b) < 1e-2


def sigmoid(a: float) -> float:
    """Compute the sigmoid function.

    Args:
    ----
        a (float): The input value.

    Returns:
    -------
        float: The sigmoid of `a`.

    """
    return (
        1.0 / (1.0 + (math.e ** (-a))) if a >= 0 else (math.e**a) / (1.0 + (math.e**a))
    )


def relu(a: float) -> float:
    """Compute the ReLU (Rectified Linear Unit) function.

    Args:
    ----
        a (float): The input value.

    Returns:
    -------
        float: The result of applying ReLU to `a`. Returns `a` if `a` is
        positive, otherwise returns 0.

    """
    return a if a > 0 else 0


def log(a: float) -> float:
    """Compute the natural logarithm of a floating-point number.

    Args:
    ----
        a (float): The input value.

    Returns:
    -------
        float: The natural logarithm of `a`.

    """
    return math.log(a)


def exp(a: float) -> float:
    """Compute the exponential of a floating-point number.

    Args:
    ----
        a (float): The input value.

    Returns:
    -------
        float: The value of `e` raised to the power of `a`.

    """
    return math.e**a


def inv(a: float) -> float:
    """Compute the multiplicative inverse of a floating-point number.

    Args:
    ----
        a (float): The input value.

    Returns:
    -------
        float: The multiplicative inverse of `a`.

    """
    return 1.0 / a


def log_back(a: float, b: float) -> float:
    """Compute the backward pass of the logarithm function with respect to the input.

    Args:
    ----
        a (float): The input value for the logarithm function.
        b (float): The gradient flowing from the output of the logarithm function.

    Returns:
    -------
        float: The gradient of the logarithm function with respect to the input `a`.

    """
    return (1.0 / a) * b


def inv_back(a: float, b: float) -> float:
    """Compute the backward pass of the inverse function with respect to the input.

    Args:
    ----
        a (float): The input value for the inverse function.
        b (float): The gradient flowing from the output of the inverse function.

    Returns:
    -------
        float: The gradient of the inverse function with respect to the input `a`.

    """
    return -(b / a**2)


def relu_back(a: float, b: float) -> float:
    """Compute the backward pass of the ReLU function with respect to the input.

    Args:
    ----
        a (float): The input value for the ReLU function.
        b (float): The gradient flowing from the output of the ReLU function.

    Returns:
    -------
        float: The gradient of the ReLU function with respect to the input `a`.
        If `a` is positive, returns `b`; otherwise returns 0.

    """
    derivative = 0 if a <= 0 else 1
    return derivative * b


def map(func: Callable, ls: Iterable) -> Iterable:
    """Apply a function to each element in an iterable.

    Args:
    ----
        func (Callable): The function to apply to each element.
        ls (Iterable): The iterable to process.

    Yields:
    ------
        Iterable: An iterable of the results of applying `func` to each element
        in `ls`.

    """
    for el in ls:
        yield func(el)


def zipWith(ls1: Iterable, ls2: Iterable, func: Callable) -> Iterable:
    """Apply a function to pairs of elements from two iterables.

    Args:
    ----
        ls1 (Iterable): The first iterable.
        ls2 (Iterable): The second iterable.
        func (Callable): The function to apply to each pair of elements.

    Yields:
    ------
        Iterable: An iterable of the results of applying `func` to each pair
        of elements from `ls1` and `ls2`.

    """
    for a, b in zip(ls1, ls2):
        yield func(a, b)


def reduce(ls: Iterable, func: Callable) -> float:
    """Reduce an iterable to a single value by iteratively applying a function.

    Args:
    ----
        ls (Iterable): The iterable to reduce.
        func (Callable): The function to apply. It should take two arguments
        and return a single value.

    Returns:
    -------
        float: The result of reducing `ls` using `func`.

    """
    current = ls[0]
    for nxt in ls[1:]:
        current = func(current, nxt)
    return current


def negList(ls: Iterable) -> Iterable:
    """Negate each element in a list.

    Args:
    ----
        ls (Iterable): The input list.

    Returns:
    -------
        Iterable: A new list where each element is negated.

    """
    return list(map(lambda x: -x, ls))


def addLists(ls1: Iterable, ls2: Iterable) -> Iterable:
    """Add corresponding elements of two lists.

    Args:
    ----
        ls1 (Iterable): The first list.
        ls2 (Iterable): The second list.

    Returns:
    -------
        Iterable: A new list where each element is the sum of the corresponding
        elements from `ls1` and `ls2`.

    """
    return list(zipWith(ls1, ls2, lambda x, y: x + y))


def sum(ls: Iterable[float]) -> float:
    """Compute the sum of all elements in a list.

    Args:
    ----
        ls (Iterable[float]): The input list of floats.

    Returns:
    -------
        float: The sum of all elements in `ls`. Returns 0 if the list is empty.

    """
    if len(ls) == 0:
        return 0
    return reduce(ls, lambda x, y: x + y)


def prod(ls: Iterable[float]) -> float:
    """Compute the product of all elements in a list.

    Args:
    ----
        ls (Iterable[float]): The input list of floats.

    Returns:
    -------
        float: The product of all elements in `ls`.

    """
    return reduce(ls, lambda x, y: x * y)
