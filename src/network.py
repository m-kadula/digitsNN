from abc import ABC, abstractmethod
from pathlib import Path
from typing import Self, NamedTuple

import numpy as np
from numpy.typing import NDArray


class ActivationFunction(ABC):

    def __call__(self, x: NDArray) -> NDArray:
        return self.f(x)

    @abstractmethod
    def f(self, x: NDArray) -> NDArray:
        pass

    @abstractmethod
    def df(self, x: NDArray) -> NDArray:
        pass


class Sigmoid(ActivationFunction):

    def f(self, x: NDArray) -> NDArray:
        return np.where(
            x >= 0,
            1 / (1 + np.exp(-x)),
            np.exp(x) / (1 + np.exp(x))
        )

    def df(self, x: NDArray) -> NDArray:
        fx = self.f(x)
        return fx * (1 - fx)


class ReLU(ActivationFunction):

    def f(self, x: NDArray) -> NDArray:
        return np.maximum(0., x)

    def df(self, x: NDArray) -> NDArray:
        return np.where(x > 0., 1., 0.)


class CostFunction(ABC):

    @abstractmethod
    def f(self, x: NDArray, expected: NDArray) -> float:
        pass

    @abstractmethod
    def df(self, x: NDArray, expected: NDArray) -> NDArray:
        pass


class Cost(CostFunction):

    def f(self, x: NDArray, expected: NDArray) -> float:
        return 0.5 * np.sum((x - expected)**2)

    def df(self, x: NDArray, expected: NDArray) -> NDArray:
        return x - expected


class NNLayer:

    def __init__(self, w: NDArray, b: NDArray):
        self.W = w
        self.b = b

    def copy(self):
        return self.__class__(self.W.copy(), self.b.copy())

    @classmethod
    def make_from_normal(cls, inp_dim: int, out_dim: int) -> Self:
        W_un = np.random.randn(out_dim, inp_dim)
        b_un = np.random.randn(out_dim)
        return cls(W_un, b_un)

    @property
    def input_dim(self) -> int:
        return self.W.shape[1]

    @property
    def output_dim(self) -> int:
        assert self.W.shape[0] == self.b.shape[0]
        return self.W.shape[0]


class NeuralNetwork:

    def __init__(self,
                 transitions: list[NNLayer],
                 activation_function: ActivationFunction,
                 cost_function: CostFunction):
        self.transitions = transitions
        self.act = activation_function
        self.cost = cost_function

    def copy(self):
        copied_transitions = [arr.copy() for arr in self.transitions]
        return self.__class__(copied_transitions, self.act, self.cost)

    @property
    def l_count(self) -> int:
        return len(self.transitions) + 1

    @classmethod
    def new_network(cls,
                    layers: list[int],
                    activation_function: ActivationFunction,
                    cost_function: CostFunction
                    ) -> Self:
        transitions = []
        for i, j in zip(layers[:-1], layers[1:]):
            transitions.append(NNLayer.make_from_normal(i, j))
        return cls(transitions, activation_function, cost_function)

    def save_to_file(self, file: Path):
        assert file.suffix == '.npz'
        assert not file.exists()
        out = {}
        for i, transition in enumerate(self.transitions):
            out[f'W{i}'] = transition.W
            out[f'b{i}'] = transition.b
        np.savez(file, **out)

    @classmethod
    def load_from_file(cls,
                       file: Path,
                       activation_function: ActivationFunction,
                       cost_function: CostFunction):
        assert file.suffix == '.npz'
        contents = np.load(file)
        transitions = []
        i = 0
        while f'W{i}' in contents.files:
            assert f'b{i}' in contents.files
            Wi = contents[f'W{i}']
            bi = contents[f'b{i}']
            transitions.append(NNLayer(Wi, bi))
            i += 1
        return cls(transitions, activation_function, cost_function)

    def __call__(self, x: NDArray) -> NDArray:
        return self.calculate(x)

    def calculate(self, x: NDArray) -> NDArray:
        a = x
        for transition in self.transitions:
            z = transition.W @ a + transition.b
            a = self.act(z)
        return a

    class BackPropagationResult(NamedTuple):
        error_w: list[NDArray]
        error_b: list[NDArray]

    def back_propagate(self, x: NDArray, expected: NDArray) -> BackPropagationResult:
        a = x
        a_list = [a]
        z_list = [a]
        for transition in self.transitions:
            z = transition.W @ a + transition.b
            a = self.act(z)
            z_list.append(z)
            a_list.append(a)

        assert len(a_list) == len(z_list) == self.l_count

        error_w = [np.empty(transition.W.shape) for transition in self.transitions]
        error_b = [np.empty(transition.b.shape) for transition in self.transitions]

        error = self.cost.df(a_list[-1], expected) * self.act.df(z_list[-1])
        error_w[-1] = error[:, np.newaxis] @ a_list[-2][np.newaxis, :]
        error_b[-1] = error
        for i in range(self.l_count - 2, 0, -1):
            error = self.transitions[i].W.T @ error * self.act.df(z_list[i])
            error_w[i - 1] = error[:, np.newaxis] @ a_list[i - 1][np.newaxis, :]
            error_b[i - 1] = error

        return self.BackPropagationResult(error_w, error_b)

    def back_propagate_batch(self, batch: tuple[NDArray, NDArray], n: float = 1.):
        assert 0 <= n <= 1

        accumulator_w = [np.zeros(transition.W.shape) for transition in self.transitions]
        accumulator_b = [np.zeros(transition.b.shape) for transition in self.transitions]

        for x, y in batch:
            weights, biases = self.back_propagate(x, y)
            for i, w in enumerate(weights):
                accumulator_w[i] += w
            for i, b in enumerate(biases):
                accumulator_b[i] += b

        for transition, weights, biases in zip(self.transitions, accumulator_w, accumulator_b):
            transition.W -= (n / len(batch)) * weights
            transition.b -= (n / len(batch)) * biases
