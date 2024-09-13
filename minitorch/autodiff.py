from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple, Protocol


# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    add = f(*vals[:arg], vals[arg] + epsilon, *vals[arg + 1 :])
    subsctract = f(*vals[:arg], vals[arg] - epsilon, *vals[arg + 1 :])
    return (add - subsctract) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None: ...  # noqa: D102

    @property
    def unique_id(self) -> int: ...  # noqa: D102

    def is_leaf(self) -> bool: ...  # noqa: D102

    def is_constant(self) -> bool: ...  # noqa: D102

    @property
    def parents(self) -> Iterable[Variable]: ...  # noqa: D102

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]: ...  # noqa: D102


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    variables = []
    variables_names = set()
    still_open = [(variable, True)]
    while len(still_open):
        (var, first_visit) = still_open[-1]
        if first_visit:
            # set first_visit = False
            still_open[-1] = (still_open[-1][0], False)
            # add previous variables with first_visit = True
            still_open.extend(zip(var.parents, [True] * len(var.parents)))
        else:
            still_open.pop()
            # push var to head of variables
            if not var.is_constant() and var.name not in variables_names:
                variables.append(var)
                variables_names.add(var.name)
    variables.reverse()
    return variables


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph to compute derivatives for the leaf nodes.

    Args:
    ----
    variable (Variable): The right-most variable (end of the computation graph) where backpropagation starts.
    deriv (Any): The derivative of the output variable with respect to itself, typically initialized to 1.0 when starting the backpropagation process.

    No return. This function writes the computed derivatives to the `derivative` attribute
    of each leaf variable through the `accumulate_derivative` method.

    """
    variables = topological_sort(variable=variable)
    derivatives = {variables[0].unique_id: deriv}

    for var in variables:
        if var.is_leaf():
            var.accumulate_derivative(derivatives[var.unique_id])
            continue

        vars_with_derivatives = var.chain_rule(derivatives[var.unique_id])
        for v, derivative in vars_with_derivatives:
            if v.unique_id in derivatives:
                derivatives[v.unique_id] += derivative
            else:
                derivatives[v.unique_id] = derivative


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:  # noqa: D102
        return self.saved_values
