import math
import random
from dataclasses import dataclass
from typing import List, Tuple


def make_pts(N: int) -> List[Tuple[float, float]]:
    """Generate N random points in a 2D plane."""
    points = []
    for _ in range(N):
        x_1 = random.random()
        x_2 = random.random()
        points.append((x_1, x_2))
    return points


@dataclass
class Graph:
    """Represents a dataset of points and their corresponding labels."""

    N: int
    X: List[Tuple[float, float]]
    y: List[int]


def simple(N: int) -> Graph:
    """Generates a simple linearly separable dataset."""
    X = make_pts(N)
    y = []
    for x_1, _ in X:
        y_label = 1 if x_1 < 0.5 else 0
        y.append(y_label)
    return Graph(N, X, y)


def diag(N: int) -> Graph:
    """Generates a dataset with a diagonal decision boundary."""
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y_label = 1 if x_1 + x_2 < 0.5 else 0
        y.append(y_label)
    return Graph(N, X, y)


def split(N: int) -> Graph:
    """Generates a dataset with points split by a vertical boundary."""
    X = make_pts(N)
    y = []
    for x_1, _ in X:
        y_label = 1 if x_1 < 0.2 or x_1 > 0.8 else 0
        y.append(y_label)
    return Graph(N, X, y)


def xor(N: int) -> Graph:
    """Generates a dataset with an XOR decision boundary."""
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y_label = 1 if (x_1 < 0.5 and x_2 > 0.5) or (x_1 > 0.5 and x_2 < 0.5) else 0
        y.append(y_label)
    return Graph(N, X, y)


def circle(N: int) -> Graph:
    """Generates a dataset with points inside and outside a circle."""
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        x1, x2 = (x_1 - 0.5, x_2 - 0.5)
        y_label = 1 if x1 * x1 + x2 * x2 > 0.1 else 0
        y.append(y_label)
    return Graph(N, X, y)


def spiral(N: int) -> Graph:
    """Generates a dataset with points forming two interleaving spirals."""

    def x(t: float) -> float:
        return t * math.cos(t) / 20.0

    def y(t: float) -> float:
        return t * math.sin(t) / 20.0

    X = [
        (x(10.0 * (float(i) / (N // 2))) + 0.5, y(10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(N // 2)
    ]
    X += [
        (y(-10.0 * (float(i) / (N // 2))) + 0.5, x(-10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(N // 2)
    ]
    y2 = [0] * (N // 2) + [1] * (N // 2)
    return Graph(N, X, y2)


# Dictionary to access different datasets
datasets = {
    "Simple": simple,
    "Diag": diag,
    "Split": split,
    "Xor": xor,
    "Circle": circle,
    "Spiral": spiral,
}
