import LSTM
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import data_load

x_dev = np.load('project_dev.npy')
y_dev = np.load('data/project_data/dev_y.npy')
# x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.1, random_state=10)

x_train = np.load('project_train.npy')
y_train = np.load('data/project_data/train_y.npy')

# x_test = np.load('K562_RR_test.npy')
# print(x_test.shape)
# y1 = np.ones(int(len(x_test)/2))
# y2 = np.zeros(int(len(x_test)/2))
# y_test = np.concatenate((y1,y2),axis=0)
# print(y_test.shape)


INPUT_SHAPE = x_train.shape[1:3]
'''KERNEL_SIZE = 5
LEARNING_RATE = 0.001
LSTM_UNITS = 32'''

LEARNING_RATE = 0.00075
KERNEL_NUMBER = 32
KERNEL_SIZE = 15
LSTM_UNITS = 64

# kernel_numbers = [32, 64, 128]
# kernel_sizes = [5, 10, 15, 20]
#
# for kernel_number in kernel_numbers:
#     for kernel_size in kernel_sizes:
#         print(kernel_number, kernel_size)
#         LSTM.three_CNN_LSTM1(x_train, y_train, x_dev, y_dev,
#                             x_dev[:10], y_dev[:10], LEARNING_RATE, INPUT_SHAPE, kernel_number, kernel_size, LSTM_UNITS,
#                              name = "three_CNN_LSTM1_{}_{}".format(kernel_number, kernel_size))



tuning, model = LSTM.three_CNN_LSTM1(x_train, y_train, x_dev, y_dev,
                    x_dev[:10], y_dev[:10], LEARNING_RATE, INPUT_SHAPE, KERNEL_NUMBER, KERNEL_SIZE, LSTM_UNITS)
# evaluate model on the first 3 examples from dev

