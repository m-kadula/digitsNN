from network import ActivationFunction, CostFunction, NNLayer, NeuralNetwork

import numpy as np
from numpy.typing import NDArray


class Sigmoid(ActivationFunction):

    def f(self, x: NDArray) -> NDArray:
        return 1 / (1 - np.exp(-x))

    def df(self, x: NDArray) -> NDArray:
        fx = self.f(x)
        return fx * (1 - fx)


class Cost(CostFunction):

    def f(self, x: NDArray, expected: NDArray) -> float:
        return 0.5 * np.sum((x - expected)**2)

    def df(self, x: NDArray, expected: NDArray) -> NDArray:
        return x - expected
