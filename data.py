import numpy as np
import tensorflow as tf
import utils.mnist_reader as mnist_reader


def dataloader():
    # -------- Read data ---------#
    train_x, train_t = mnist_reader.load_mnist('data/', kind='train')
    test_x, test_t = mnist_reader.load_mnist('data/', kind='t10k')
    # ------ Preprocess data -----#
    # x_train = train_x.reshape(train_x.shape[0], 28, 28, 1).astype('float32')
    x_train = train_x
    x_train_norm = x_train / 255.0
    # x_test = test_x.reshape(test_x.shape[0], 28, 28, 1).astype('float32')
    x_test = test_x
    x_test_norm = x_test / 255.0
    print(np.shape(train_x), np.shape(train_t))
    print(np.shape(x_train_norm), np.shape(train_t))
    return x_train_norm, train_t, x_test_norm, test_t

def define_dataset(tr_x, tr_y, batch_size, buffer_size):
    # dataset1 = tf.data.Dataset.from_tensor_slices(tr_x)
    # dataset2 = tf.data.Dataset.from_tensor_slices(tr_y)
    # dataset3 = tf.data.Dataset.zip((dataset1, dataset2))
    # print(dataset1.output_types)  # ==> "tf.float32"
    # print(dataset1.output_shapes)  # ==> "(10,)"
    #
    # print(dataset2.output_types)  # ==> "(tf.float32, tf.int32)"
    # print(dataset2.output_shapes)  # ==> "((), (100,))"
    #
    # print(dataset3.output_types)  # ==> (tf.float32, (tf.float32, tf.int32))
    # print(dataset3.output_shapes)  # ==> "(10, ((), (100,)))"

    dataset = tf.data.Dataset.from_tensor_slices(
        {"x": tr_x,
         "y": tr_y})
    print(dataset.output_types)  # ==> "{'a': tf.float32, 'b': tf.int32}"
    print(dataset.output_shapes)  # ==> "{'a': (), 'b': (100,)}"
    dataset = dataset.repeat()
    dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    return next_element, iterator