from random import shuffle

import numpy as np

from data import x_train, x_test, y_train, y_test


class NeuralNetwork():
    """
    this is a basic class of neural network
    """
    eta = 5

    def __init__(self, layers):
        # first we assume this network contains
        # only 3 layers: input, hidden, out
        self.n_input = layers[0]
        self.n_hidden = layers[1]
        self.n_out = layers[2]
        # initial random weight
        self.w_in2hidden = np.random.randn(self.n_hidden, self.n_input + 1)
        self.w_hidden2out = np.random.randn(10, self.n_hidden + 1)

    def train(self, x_train, y_train):
        """
        :param x_train: 2d array, features
        :param y_train: 1d array, labels of x_train
        :return: None
        """
        n_samples, n_features = x_train.shape
        y_vec = np.zeros((n_samples, 10))
        y_vec[np.arange(n_samples), y_train] = 1

        orders = list(range(n_samples))
        shuffle(orders)

        N = 20
        n_batch = 0
        while (N * n_batch <= n_samples):
            n_batch += 1
            if n_batch % 50 == 0:
                print("processing:{0:.1f}%".format(100 * n_batch * N / n_samples), end="\r")
            batch = orders[N * n_batch: N * n_batch + N]
            errw_out2hidden = np.zeros((10, self.n_hidden + 1))
            errw_hidden2in = np.zeros((self.n_hidden, n_features + 1))
            for i in batch:
                y_sample = y_vec[orders[i]]
                sample = x_train[orders[i]]
                # forward
                # sample add 1 at the tail
                sample_add1 = np.hstack((sample, [1])).T
                u_hidden = np.dot(self.w_in2hidden, sample_add1)
                a_hidden = self.sigmoid(u_hidden)
                a_hidden_add1 = np.hstack((a_hidden, [1]))
                u_out = np.dot(self.w_hidden2out, a_hidden_add1)
                a_out = self.sigmoid(u_out)
                cost_dev = a_out - y_sample
                # back
                err_out = cost_dev * self.sigmoid_prime(u_out)
                errw_out2hidden += np.dot(err_out[:, np.newaxis], a_hidden_add1[np.newaxis, :])

                err_hidden = np.dot(self.w_hidden2out.T, err_out)
                err_hidden = err_hidden[:-1]
                err_hidden *= self.sigmoid_prime(u_hidden)
                errw_hidden2in += np.dot(err_hidden[:, np.newaxis], sample_add1[np.newaxis, :])
            # update weight
            self.w_in2hidden -= self.eta * errw_hidden2in / N
            self.w_hidden2out -= self.eta * errw_out2hidden / N


    def sigmoid(self, x):
        return 1. / (1 + np.exp(-x))

    def sigmoid_prime(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def predict(self, x_test):
        """
        predict the result of x_test
        :param x_test: 2d array, features
        :return: 1d array, the prediction of x_test
        """
        y_test = []
        for sample in x_test:
            sample_add1 = np.hstack((sample, [1]))
            u_hidden = np.dot(self.w_in2hidden, sample_add1)
            a_hidden = self.sigmoid(u_hidden)
            a_hidden_add1 = np.hstack((a_hidden, [1]))
            u_out = np.dot(self.w_hidden2out, a_hidden_add1)
            y_out = self.sigmoid(u_out)
            y_test.append(y_out)
        return y_test


if __name__ == '__main__':
    nn = NeuralNetwork([784, 30, 10])

    n_round = 30
    for r in range(n_round):
        print("round:", r)
        nn.train(x_train, y_train)
        predicty = nn.predict(x_test)
        print("result:", np.sum(np.argmax(predicty, axis=1) == y_test))
