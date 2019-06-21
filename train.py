
import numpy as np
import tensorflow as tf
import utils.mnist_reader as mnist_reader
from keras.utils import np_utils

if __name__ == "__main__":
    # -------- Read data ---------#
    train_x, train_t = mnist_reader.load_mnist('data/', kind='train')
    test_x, test_t = mnist_reader.load_mnist('data/', kind='t10k')
    # ------ Preprocess data -----#
    x_train = train_x.reshape(train_x.shape[0], 28, 28, 1).astype('float32')
    x_train_norm = x_train / 255.0
    t_train_onehot = np_utils.to_categorical(train_t)
    x_test = test_x.reshape(test_x.shape[0], 28, 28, 1).astype('float32')
    x_test_norm = x_test / 255.0
    test_one_hot = np_utils.to_categorical(test_t)
    print(np.shape(train_x), np.shape(train_t))
    print(np.shape(x_train), np.shape(t_train_onehot))

    epoch = 500
    LR = 0.001
