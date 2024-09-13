from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence, Tuple, Type, Union

import numpy as np

from .autodiff import Context, Variable, backpropagate, central_difference
from .scalar_functions import (
    EQ,
    LT,
    Add,
    Exp,
    Inv,
    Log,
    Mul,
    Neg,
    ReLU,
    ScalarFunction,
    Sigmoid,
)

ScalarLike = Union[float, int, "Scalar"]


@dataclass
class ScalarHistory:
    """`ScalarHistory` stores the history of `Function` operations that was
    used to construct the current Variable.

    Attributes
    ----------
    last_fn : Optional[Type[ScalarFunction]]
        The last Function that was called.
    ctx : Optional[Context]
        The context for that Function.
    inputs : Sequence[Scalar]
        The inputs that were given when `last_fn.forward` was called.

    """

    last_fn: Optional[Type[ScalarFunction]] = None
    ctx: Optional[Context] = None
    inputs: Sequence[Scalar] = ()


_var_count = 0


class Scalar:
    """A reimplementation of scalar values for autodifferentiation
    tracking. Scalar Variables behave as close as possible to standard
    Python numbers while also tracking the operations that led to the
    number's creation. They can only be manipulated by
    `ScalarFunction`.
    """

    history: Optional[ScalarHistory]
    derivative: Optional[float]
    data: float
    unique_id: int
    name: str

    def __init__(
        self,
        v: float,
        back: ScalarHistory = ScalarHistory(),
        name: Optional[str] = None,
    ):
        """Initialize a Scalar instance.

        Args:
        ----
            v (float): The initial value of the scalar.
            back (ScalarHistory, optional): The history of operations leading to this scalar.
            name (Optional[str], optional): An optional name for the scalar.

        """
        global _var_count
        _var_count += 1
        self.unique_id = _var_count
        self.data = float(v)
        self.history = back
        self.derivative = None
        if name is not None:
            self.name = name
        else:
            self.name = str(self.unique_id)

    def __repr__(self) -> str:
        """Return a string representation of the Scalar instance.

        Returns
        -------
            str: The string representation of the Scalar.

        """
        return f"Scalar({self.data})"

    def __mul__(self, b: ScalarLike) -> Scalar:
        """Multiply this scalar by another value.

        Args:
        ----
            b (ScalarLike): The value to multiply with.

        Returns:
        -------
            Scalar: The result of the multiplication.

        """
        return Mul.apply(self, b)

    def __truediv__(self, b: ScalarLike) -> Scalar:
        """Divide this scalar by another value.

        Args:
        ----
            b (ScalarLike): The value to divide by.

        Returns:
        -------
            Scalar: The result of the division.

        """
        return Mul.apply(self, Inv.apply(b))

    def __rtruediv__(self, b: ScalarLike) -> Scalar:
        """Divide another value by this scalar.

        Args:
        ----
            b (ScalarLike): The value to be divided.

        Returns:
        -------
            Scalar: The result of the division.

        """
        return Mul.apply(b, Inv.apply(self))

    def __bool__(self) -> bool:
        """Return the boolean value of this scalar.

        Returns
        -------
            bool: The boolean value of the scalar.

        """
        return bool(self.data)

    def __add__(self, b: ScalarLike) -> Scalar:
        """Add this scalar to another value.

        Args:
        ----
            b (ScalarLike): The value to add.

        Returns:
        -------
            Scalar: The result of the addition.

        """
        return Add.apply(self, b)

    def __radd__(self, b: ScalarLike) -> Scalar:
        """Add another value to this scalar.

        Args:
        ----
            b (ScalarLike): The value to add.

        Returns:
        -------
            Scalar: The result of the addition.

        """
        return self + b

    def __rmul__(self, b: ScalarLike) -> Scalar:
        """Multiply another value by this scalar.

        Args:
        ----
            b (ScalarLike): The value to multiply.

        Returns:
        -------
            Scalar: The result of the multiplication.

        """
        return self * b

    def __lt__(self, b: ScalarLike) -> Scalar:
        """Compare if this scalar is less than another value.

        Args:
        ----
            b (ScalarLike): The value to compare with.

        Returns:
        -------
            Scalar: The result of the comparison.

        """
        return LT.apply(self, b)

    def __eq__(self, b: ScalarLike) -> Scalar:
        """Compare if this scalar is equal to another value.

        Args:
        ----
            b (ScalarLike): The value to compare with.

        Returns:
        -------
            Scalar: The result of the comparison.

        """
        return EQ.apply(self, b)

    def __sub__(self, b: ScalarLike) -> Scalar:
        """Subtract another value from this scalar.

        Args:
        ----
            b (ScalarLike): The value to subtract.

        Returns:
        -------
            Scalar: The result of the subtraction.

        """
        return Add.apply(self, Neg.apply(b))

    def __rsub__(self, b: ScalarLike) -> Scalar:
        """Subtract this scalar from another value.

        Args:
        ----
            b (ScalarLike): The value to subtract from.

        Returns:
        -------
            Scalar: The result of the subtraction.

        """
        return self - b

    def __neg__(self) -> Scalar:
        """Negate this scalar.

        Returns
        -------
            Scalar: The result of negating the scalar.

        """
        return Neg.apply(self)

    def log(self) -> Scalar:
        """Compute the logarithm of this scalar.

        Returns
        -------
            Scalar: The result of the logarithm.

        """
        return Log.apply(self)

    def exp(self) -> Scalar:
        """Compute the exponential of this scalar.

        Returns
        -------
            Scalar: The result of the exponential.

        """
        return Exp.apply(self)

    def sigmoid(self) -> Scalar:
        """Compute the sigmoid of this scalar.

        Returns
        -------
            Scalar: The result of the sigmoid.

        """
        return Sigmoid.apply(self)

    def relu(self) -> Scalar:
        """Compute the ReLU of this scalar.

        Returns
        -------
            Scalar: The result of the ReLU.

        """
        return ReLU.apply(self)

    def accumulate_derivative(self, x: Any) -> None:
        """Add `val` to the derivative accumulated on this variable.
        Should only be called during autodifferentiation on leaf variables.

        Args:
        ----
            x (Any): Value to be accumulated.

        """
        assert self.is_leaf(), "Only leaf variables can have derivatives."
        if self.derivative is None:
            self.__setattr__("derivative", 0.0)
        self.__setattr__("derivative", self.derivative + x)

    def is_leaf(self) -> bool:
        """Check if this variable is a leaf variable (i.e., created by the user).

        Returns
        -------
            bool: True if this is a leaf variable, otherwise False.

        """
        return self.history is not None and self.history.last_fn is None

    def is_constant(self) -> bool:
        """Check if this variable is constant (i.e., has no history).

        Returns
        -------
            bool: True if this variable is constant, otherwise False.

        """
        return self.history is None

    @property
    def parents(self) -> Iterable[Variable]:
        """Get the parent variables that were used to compute this variable.

        Returns
        -------
            Iterable[Variable]: The parent variables.

        """
        assert self.history is not None
        return self.history.inputs

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Compute the chain rule for backpropagation.

        Args:
        ----
            d_output (Any): The gradient of the output.

        Returns:
        -------
            Iterable[Tuple[Variable, Any]]: A sequence of tuples where each tuple contains
            a parent variable and its corresponding gradient.

        """
        h = self.history
        assert h is not None
        assert h.last_fn is not None
        assert h.ctx is not None

        derivatives = h.last_fn._backward(h.ctx, d_output)
        variables = h.inputs

        return zip(variables, derivatives)

    def backward(self, d_output: Optional[float] = None) -> None:
        """Call autodiff to fill in the derivatives for the history of this object.

        Args:
        ----
            d_output (Optional[float], optional): Starting derivative to backpropagate through
            the model (typically left out, and assumed to be 1.0).

        """
        if d_output is None:
            d_output = 1.0
        backpropagate(self, d_output)


def derivative_check(f: Any, *scalars: Scalar) -> None:
    """Check that autodiff works on a Python function and assert correctness of derivatives.

    Args:
    ----
        f (Any): Function from n-scalars to 1-scalar.
        *scalars (Scalar): n input scalar values.

    Raises:
    ------
        AssertionError: If the computed derivative does not match the central difference approximation.

    """
    out = f(*scalars)
    out.backward()

    err_msg = """
Derivative check at arguments f(%s) and received derivative f'=%f for argument %d,
but was expecting derivative f'=%f from central difference."""
    for i, x in enumerate(scalars):
        check = central_difference(f, *scalars, arg=i)
        print(str([x.data for x in scalars]), x.derivative, i, check)
        assert x.derivative is not None
        np.testing.assert_allclose(
            x.derivative,
            check.data,
            1e-2,
            1e-2,
            err_msg=err_msg
            % (str([x.data for x in scalars]), x.derivative, i, check.data),
        )
