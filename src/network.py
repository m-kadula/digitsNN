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


class CostFunction(ABC):

    @abstractmethod
    def f(self, x: NDArray, expected: NDArray) -> NDArray:
        pass

    @abstractmethod
    def df(self, x: NDArray, expected: NDArray) -> NDArray:
        pass


class NNLayer:

    def __init__(self, w: NDArray, b: NDArray):
        self.W = w
        self.b = b

    def copy(self):
        return self.__class__(self.W.copy(), self.b.copy())

    @staticmethod
    def _normalise(a: NDArray) -> NDArray:
        return 2. * (a - np.min(a)) / np.ptp(a) - 1

    @classmethod
    def make_from_normal(cls, inp_dim: int, out_dim: int, expected: float = 0, deviation: float = 0.5) -> Self:
        W_un = np.random.normal(expected, deviation, (out_dim, inp_dim))
        b_un = np.random.normal(expected, deviation, (out_dim,))
        return cls(cls._normalise(W_un), cls._normalise(b_un))

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
                    cost_function: CostFunction,
                    expected: float = 0,
                    deviation: float = 0.5) -> Self:
        transitions = []
        for i, j in zip(layers[:-1], layers[1:]):
            transitions.append(NNLayer.make_from_normal(i, j, expected, deviation))
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
            Wi = contents.files[f'W{i}']
            bi = contents.files[f'b{i}']
            transitions.append(NNLayer(Wi, bi))
            i += 1
        return cls(transitions, activation_function, cost_function)

    def __call__(self, x: NDArray) -> NDArray:
        return self.calculate(x)

    def calculate(self, x: NDArray) -> NDArray:
        a = self.act(x)
        for transition in self.transitions:
            z = transition.W @ a + transition.b
            a = self.act(z)
        return a

    class BackPropagationResult(NamedTuple):
        grad: list[NDArray]
        a: list[NDArray]

    def back_propagate(self, x: NDArray, expected: NDArray) -> BackPropagationResult:
        z_list = [None] * self.l_count
        a_list = [None] * self.l_count

        a = self.act(x)
        z_list[0] = x
        a_list[0] = a
        for i, transition in enumerate(self.transitions, start=1):
            z = transition.W @ a + transition.b
            a = self.act(z)
            z_list[i] = z
            a_list[i] = a

        error_list = [None] * self.l_count
        error_list[-1] = self.cost.df(a_list[-1], expected)
        for i in reversed(range(1, self.l_count - 1)):
            transition = self.transitions[i + 1]
            error_list[i] = (transition.W.T @ error_list[i + 1]) * self.act.df(z_list[i])

        return self.BackPropagationResult(grad=error_list[1:], a=a_list)
