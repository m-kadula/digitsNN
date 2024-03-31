from abc import ABC, abstractmethod
from pathlib import Path
from typing import Self, Iterable

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

    def __call__(self, x: NDArray, expected: NDArray) -> float:
        return self.f(x, expected)

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
    def get_random(cls, inp_dim: int, out_dim: int) -> Self:
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
        assert len(transitions) >= 1
        for t1, t2 in zip(transitions[:-1], transitions[1:]):
            assert t1.output_dim == t2.input_dim
        self.transitions = transitions
        self.act = activation_function
        self.cost = cost_function

    @property
    def input_dim(self) -> int:
        return self.transitions[0].input_dim

    @property
    def output_dim(self) -> int:
        return self.transitions[-1].output_dim

    @property
    def l_count(self) -> int:
        return len(self.transitions) + 1

    @classmethod
    def new_network(cls,
                    layers: Iterable[int],
                    activation_function: ActivationFunction,
                    cost_function: CostFunction
                    ) -> Self:
        transitions = []
        for i, j in zip(layers[:-1], layers[1:]):
            transitions.append(NNLayer.get_random(i, j))
        return cls(transitions, activation_function, cost_function)

    def copy(self):
        copied_transitions = [arr.copy() for arr in self.transitions]
        return self.__class__(copied_transitions, self.act, self.cost)

    def save_to_file(self, file: Path | str):
        if isinstance(file, str):
            file = Path(file).resolve()
        assert file.suffix == '.npz'
        assert not file.exists()
        out = {}
        for i, transition in enumerate(self.transitions):
            out[f'W{i}'] = transition.W
            out[f'b{i}'] = transition.b
        np.savez(file, **out)

    @classmethod
    def load_from_file(cls,
                       file: Path | str,
                       activation_function: ActivationFunction,
                       cost_function: CostFunction):
        if isinstance(file, str):
            file = Path(file).resolve()
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

    def backprop(self, x: NDArray, expected: NDArray) -> tuple[list[NDArray], list[NDArray]]:
        assert len(x.shape) == len(expected.shape) == 2 and x.shape[0] == expected.shape[0]

        a = x
        a_list = [a]
        z_list = []
        for transition in self.transitions:
            z = np.einsum('kj,ij->ik', transition.W, a) + transition.b[np.newaxis, :]
            a = self.act(z)
            z_list.append(z)
            a_list.append(a)

        error_w = [np.empty((x.shape[0], *transition.W.shape)) for transition in self.transitions]
        error_b = [np.empty((x.shape[0], *transition.b.shape)) for transition in self.transitions]

        error = self.cost.df(a_list[-1], expected) * self.act.df(z_list[-1])
        error_w[-1] = np.einsum('ij,il->ijl', error, a_list[-2])
        error_b[-1] = error
        for i in range(self.l_count - 2, 0, -1):
            error = np.einsum('kj,ij->ik', self.transitions[i].W.T, error) * self.act.df(z_list[i - 1])
            error_w[i - 1] = np.einsum('ij,il->ijl', error, a_list[i - 1])
            error_b[i - 1] = error

        sum_error_w = [np.sum(cw, axis=0) for cw in error_w]
        sum_error_b = [np.sum(cb, axis=0) for cb in error_b]
        return sum_error_w, sum_error_b

    def update_network(self, weights: list[NDArray], biases: list[NDArray], scale: float, batch_size: int):
        multiplier = scale / batch_size
        for transition, w, b in zip(self.transitions, weights, biases, strict=True):
            transition.W -= multiplier * w
            transition.b -= multiplier * b

    def backprop_and_apply(self, x: NDArray, expected: NDArray, scale: float):
        w, b = self.backprop(x, expected)
        self.update_network(w, b, scale, x.shape[0])
