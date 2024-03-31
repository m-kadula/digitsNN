from abc import ABC, abstractmethod
from pathlib import Path
from typing import Self, Iterable

import numpy as np
from numpy.typing import NDArray


class ActivationFunction(ABC):
    """Base class for activation functions"""

    def __call__(self, x: NDArray) -> NDArray:
        return self.f(x)

    @abstractmethod
    def f(self, x: NDArray) -> NDArray:
        """f: M(i,j) -> M(i,j) where M[i, :] - vectors, M[:, j] - vector body"""
        pass

    @abstractmethod
    def df(self, x: NDArray) -> NDArray:
        """(dy/dx): M(i,j) -> M(i,j) where M[i, :] - vectors, M[:, j] - vector body"""
        pass


class Sigmoid(ActivationFunction):
    """Sigmoid activation function"""

    def f(self, x: NDArray) -> NDArray:
        return np.where(
            x >= 0,
            1 / (1 + np.exp(-x)),
            np.exp(x) / (1 + np.exp(x))
        )

    def df(self, x: NDArray) -> NDArray:
        fx = self.f(x)
        return fx * (1 - fx)


class CostFunction(ABC):

    def __call__(self, x: NDArray, expected: NDArray) -> float:
        return self.f(x, expected)

    @abstractmethod
    def f(self, x: NDArray, expected: NDArray) -> float:
        """f: M(i,j), M(i,j) -> R where M[i, :] - vectors, M[:, j] - vector body"""
        pass

    @abstractmethod
    def df(self, x: NDArray, expected: NDArray) -> NDArray:
        """(dy/dx): M(i,j), M(i,j) -> M(i,j) where M[i, :] - vectors, M[:, j] - vector body"""
        pass


class Cost(CostFunction):
    """Standard cost function"""

    def f(self, x: NDArray, expected: NDArray) -> float:
        return 0.5 * np.sum((x - expected)**2)

    def df(self, x: NDArray, expected: NDArray) -> NDArray:
        return x - expected


class NNLayer:
    """Single layer of a neural network (f(x) = Wx + b)"""

    def __init__(self, w: NDArray, b: NDArray):
        """Create a new NNLayer with given weights and biases"""
        self.W = w
        self.b = b

    def copy(self):
        """Return a deep copy of self"""
        return self.__class__(self.W.copy(), self.b.copy())

    @classmethod
    def get_random(cls, inp_dim: int, out_dim: int) -> Self:
        """Generate a new NNLayer with random weights and biases from N(0, 1)"""
        W_un = np.random.randn(out_dim, inp_dim)
        b_un = np.random.randn(out_dim)
        return cls(W_un, b_un)

    @property
    def input_dim(self) -> int:
        """Input dimension for function f(x) = Wx + b"""
        return self.W.shape[1]

    @property
    def output_dim(self) -> int:
        """Output dimension for function f(x) = Wx + b"""
        assert self.W.shape[0] == self.b.shape[0]
        return self.W.shape[0]


class NeuralNetwork:
    """Neural network training and evaluation class"""

    def __init__(self,
                 transitions: list[NNLayer],
                 activation_function: ActivationFunction,
                 cost_function: CostFunction):
        """Create a new neural network with given transitions, activation function and cost function"""
        if len(transitions) < 1:
            raise ValueError("At least one transition is required")
        for t1, t2 in zip(transitions[:-1], transitions[1:]):
            if t1.output_dim != t2.input_dim:
                raise ValueError(f"Transitions {t1} and {t2} are incompatible ({t1.output_dim=} != {t2.input_dim=})")
        self.transitions = transitions
        self.act = activation_function
        self.cost = cost_function

    @property
    def input_dim(self) -> int:
        """Input dimension for the network"""
        return self.transitions[0].input_dim

    @property
    def output_dim(self) -> int:
        """Output dimension for the network"""
        return self.transitions[-1].output_dim

    @property
    def l_count(self) -> int:
        """Number of layers in the network"""
        return len(self.transitions) + 1

    @classmethod
    def new_network(cls,
                    layers: Iterable[int],
                    activation_function: ActivationFunction,
                    cost_function: CostFunction
                    ) -> Self:
        """Create a new neural network with random weights and biases from N(0, 1)"""
        transitions = []
        for i, j in zip(layers[:-1], layers[1:]):
            transitions.append(NNLayer.get_random(i, j))
        return cls(transitions, activation_function, cost_function)

    def copy(self):
        """Return a deep copy of self"""
        copied_transitions = [arr.copy() for arr in self.transitions]
        return self.__class__(copied_transitions, self.act, self.cost)

    def save_to_file(self, file: Path | str, overwrite: bool = False):
        """Save the network to a file in .npz format"""
        if isinstance(file, str):
            file = Path(file).resolve()
        if file.suffix != '.npz':
            raise ValueError("File must have .npz extension")
        if not overwrite and file.exists():
            raise FileExistsError(f"File {file} already exists")
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
        """Load the network from a file in .npz format"""
        if isinstance(file, str):
            file = Path(file).resolve()
        if file.suffix != '.npz':
            raise ValueError("File must have .npz extension")
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
        """Evaluate the network on input x"""
        return self.calculate(x)

    def calculate(self, x: NDArray) -> NDArray:
        """Evaluate the network on input x"""
        a = x
        for transition in self.transitions:
            z = transition.W @ a + transition.b
            a = self.act(z)
        return a

    def backprop(self, x: NDArray, expected: NDArray) -> tuple[list[NDArray], list[NDArray]]:
        """
        Calculate gradients for weights and biases using backpropagation algorithm in batch mode.
        x and y: M(i,j) where i - vector index, j - vector body
        """
        if not (len(x.shape) == len(expected.shape) == 2 and x.shape[0] == expected.shape[0]):
            raise ValueError("x and expected must be 2D arrays with the same number of rows")

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
        """Update the network weights and biases using given gradients and scale factor."""
        multiplier = scale / batch_size
        for transition, w, b in zip(self.transitions, weights, biases, strict=True):
            transition.W -= multiplier * w
            transition.b -= multiplier * b

    def backprop_and_apply(self, x: NDArray, expected: NDArray, scale: float):
        """
        Calculate gradients and update the network weights and biases using
        backpropagation algorithm and given scale factor.
        """
        w, b = self.backprop(x, expected)
        self.update_network(w, b, scale, x.shape[0])
