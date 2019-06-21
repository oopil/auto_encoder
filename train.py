import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from keras.utils import np_utils
from data import *
from model import *

def parse_args() -> argparse:
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0', type=str)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # ----------------- data load part ---------------- #
    tr_x, tr_y, tst_x, tst_y = dataloader()

    # ----------------- graph build part ---------------- #
    next_element, iterator = define_dataset(tr_x, tr_y, 1000, 1000)
    x = next_element['x']
    y_gt = tf.one_hot(next_element['y'], 10)
    print(x, y_gt)
    net = neuralnet()
    y_pred = net.build(inputs=x, labels=y_gt, ch=1024)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_gt, logits=y_pred)
    loss = tf.reduce_mean(cross_entropy)
    with tf.variable_scope('optimizer'):
        rate = tf.placeholder(dtype=tf.float32)
        optimizer = tf.train.AdamOptimizer(rate)
        train_step = optimizer.minimize(loss)
        print(y_pred.shape)
        print(y_pred)
        correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_gt, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Summarize
    tf.summary.scalar("loss", loss)
    tf.summary.scalar("accuracy", accuracy)
    merged_summary = tf.summary.merge_all()

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    epochs = 1000
    lr = 0.001
    for i in range(epochs):
        _, loss_tr, accur_tr = sess.run((train_step, loss, accuracy),
                                        feed_dict={rate:lr})
        print(i,loss_tr, accur_tr)
    assert False

    tr_y_hot = np_utils.to_categorical(tr_y)
    tst_y_hot = np_utils.to_categorical(tst_y)

    epoch = 500
    LR = 0.001

    ch = args.ch
    batch = args.batch_size  # 10
    dropout_prob = 0.5
    epochs = args.epoch
    is_mask = args.mask
    print_freq = 5
    learning_rate = args.lr
    '''
        model building parts
    '''
    class_num = 10
    img_size = 28
    s1, s2 = img_size, img_size
    images = tf.placeholder(tf.float32, (None, s1, s2, 1), name='inputs')
    # lh, rh = tf.split(images, [patch_size, patch_size], 1)
    y_gt = tf.placeholder(tf.float32, (None, 10))
    keep_prob = tf.placeholder(tf.float32)

    network = None
    if args.network == 'simple':
        network = Simple
    else:
        assert False
    assert network != None

    # patch_num = 2
    initializer = tf.contrib.layers.xavier_initializer()
    # initializer = tf.truncated_normal_initializer

    my_model = network(weight_initializer=initializer,
                       activation=tf.nn.relu,
                       class_num=class_num,
                       patch_size=s1)
    y = my_model.model(images)
    # %%
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_gt, logits=y)
    loss = tf.reduce_mean(cross_entropy)

    with tf.name_scope('learning_rate_decay'):
        start_lr = learning_rate
        global_step = tf.Variable(0, trainable=False)
        total_learning = epochs
        # lr = tf.train.exponential_decay(start_lr, global_step,total_learning,0.99999, staircase=True)
        lr = tf.train.exponential_decay(start_lr, global_step, decay_steps=epochs // 100, decay_rate=.96,
                                        staircase=True)

    with tf.variable_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(lr)
        train_step = optimizer.minimize(loss)
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_gt, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Summarize
    tf.summary.scalar("loss", loss)
    tf.summary.scalar("accuracy", accuracy)
    merged_summary = tf.summary.merge_all()

    # model_vars = tf.trainable_variables()
    # tf.contrib.slim.model_analyzer.analyze_vars(model_vars, print_info=True)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        pass