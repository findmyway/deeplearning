import os

import numpy as np

curpath = os.path.dirname(__file__)
x_train = np.load(curpath + '/x_train.npy')
x_test = np.load(curpath + '/x_test.npy')
y_train = np.load(curpath + '/y_train.npy')
y_test = np.load(curpath + '/y_test.npy')