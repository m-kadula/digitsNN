from network import Sigmoid, Cost, NeuralNetwork

import numpy as np
from numpy.typing import NDArray


def itoa(i: int) -> NDArray:
    arr = np.zeros((10,))
    arr[i] = 1.
    return arr


def main():
    nn = NeuralNetwork.new_network([64, 16, 16, 10], Sigmoid(), Cost())
    from sklearn import datasets
    digits = datasets.load_digits()
    data = []
    for d_set in digits['data'] / 16:
        data.append(d_set)
    target = []
    for t_set in digits['target']:
        target.append(itoa(t_set))
    batch = list(zip(data[:1000], target[:1000]))
    minibatches = [batch[i:i+50] for i in range(0, len(batch), 50)]
    print('starting...')
    for _ in range(1000):
        for mini_batch in minibatches:
            nn.back_propagate_batch(mini_batch)
    print('finished!')
    acc = 0
    cost = Cost()
    dset = list(zip(data[:1000], target[:1000]))
    for x, y in dset:
        outcome = nn.calculate(x)
        print(outcome, y)
        if cost.f(outcome, y) < 0.1:
            acc += 1
    print(f'finished with {acc} out of {len(dset)} correct.')


if __name__ == '__main__':
    main()
