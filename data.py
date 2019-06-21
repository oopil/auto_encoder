import numpy as np
import utils.mnist_reader as mnist_reader

def dataloader():
    # -------- Read data ---------#
    train_x, train_t = mnist_reader.load_mnist('data/', kind='train')
    test_x, test_t = mnist_reader.load_mnist('data/', kind='t10k')
    # ------ Preprocess data -----#
    x_train = train_x.reshape(train_x.shape[0], 28, 28, 1).astype('float32')
    x_train_norm = x_train / 255.0
    x_test = test_x.reshape(test_x.shape[0], 28, 28, 1).astype('float32')
    x_test_norm = x_test / 255.0
    print(np.shape(train_x), np.shape(train_t))
    print(np.shape(x_train_norm), np.shape(train_t))
    return x_train_norm, train_t, x_test_norm, test_t