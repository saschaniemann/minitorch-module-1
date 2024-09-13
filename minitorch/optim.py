from typing import Sequence

from .module import Parameter
from .scalar import Scalar


class Optimizer:
    """Base class for all optimizers.

    Args:
    ----
        parameters (Sequence[Parameter]): A sequence of `Parameter` objects to be optimized.

    """

    def __init__(self, parameters: Sequence[Parameter]):
        """Initialize the Optimizer.

        Args:
        ----
            parameters (Sequence[Parameter]): Parameters to optimize.

        """
        self.parameters = parameters


class SGD(Optimizer):
    """Stochastic Gradient Descent (SGD) optimizer.

    Args:
    ----
        parameters (Sequence[Parameter]): A sequence of `Parameter` objects to be optimized.
        lr (float, optional): Learning rate for the optimizer. Default is 1.0.

    """

    def __init__(self, parameters: Sequence[Parameter], lr: float = 1.0):
        """Initialize the SGD optimizer.

        Args:
        ----
            parameters (Sequence[Parameter]): Parameters to optimize.
            lr (float, optional): Learning rate for the optimizer. Default is 1.0.

        """
        super().__init__(parameters)
        self.lr = lr

    def zero_grad(self) -> None:
        """Reset the gradients of all parameters to zero.

        Sets the `derivative` or `grad` attribute of each parameter's value to `None`.
        """
        for p in self.parameters:
            if p.value is None:
                continue
            if hasattr(p.value, "derivative"):
                if p.value.derivative is not None:
                    p.value.derivative = None
            if hasattr(p.value, "grad"):
                if p.value.grad is not None:
                    p.value.grad = None

    def step(self) -> None:
        """Perform a single optimization step.

        Updates the parameters based on their gradients (`derivative` or `grad`) and the learning rate.
        """
        for p in self.parameters:
            if p.value is None:
                continue
            if hasattr(p.value, "derivative"):
                if p.value.derivative is not None:
                    p.update(Scalar(p.value.data - self.lr * p.value.derivative))
            elif hasattr(p.value, "grad"):
                if p.value.grad is not None:
                    p.update(p.value - self.lr * p.value.grad)
