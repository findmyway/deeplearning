import numpy as np

from data import x_train, x_test, y_train, y_test


class NeuralNetwork():
    """
    this is a basic class of neural network
    """

    def __init__(self, layers):
        # first we assume this network contains
        # only 3 layers: input, hidden, out
        self.n_input = layers[0]
        self.n_hidden = layers[0]
        self.n_out = layers[0]

    def train(self, x_train, y_train):
        """
        :param x_train: 2d array, features
        :param y_train: 1d array, labels of x_train
        :return: None
        """
        pass

    def predict(self, x_test):
        """
        predict the result of x_test
        :param x_test: 2d array, features
        :return: 1d array, the prediction of x_test
        """
        y_test = []
        for sample in x_test:
            # todo
            # gen prediction of each sample
            # y_test.appen()
            pass
        return y_test


if __name__ == '__main__':
    nn = NeuralNetwork([700, 30, 10])
    nn.train(x_train, y_train)
    predicty = nn.predict(x_test)
    print(np.sum(predicty == y_test))
