import numpy as np
import tensorflow as tf
##################################################################################
# Custom Operation
##################################################################################
def conv2d(x, ch, ks, st, padding='same', activ=tf.nn.relu):
    return tf.layers.conv2d(x,
                            filters=ch,
                            kernel_size=ks,
                            strides=st,
                            padding=padding,
                            activation=activ)

def deconv2d(x, ch, ks, st, padding='same', activ=tf.nn.relu):
    return tf.layers.conv2d_transpose(x,
                            filters=ch,
                            kernel_size=ks,
                            strides=st,
                            padding=padding,
                            activation=activ)

class encoder:
    def __init__(self):
        pass

    def conv2d(self, x, ch, ks, st, padding='same', activ=tf.nn.relu):
        return tf.layers.conv2d(inputs=x,
                                filters=ch,
                                kernel_size=ks,
                                padding=padding,
                                activation=activ,
                                strides=st)
    #                                kernel_initializer=tf.,
    #                             bias_initializer=self.weight_init

    def build(self, inputs, ch_in, ch=64):
        activ = tf.nn.relu
        ks = 4
        st = 2
        with tf.variable_scope("encoder", reuse=False):
            x = inputs
            for i in range(2):
                x = conv2d(x, ch, ks, 1, 'same', activ)
                x = conv2d(x, ch, ks, 1, 'same', activ)
                x = conv2d(x, ch, ks, st, 'same', activ)
                ch *= 2

            # x = conv2d(x, ch, ks, 1, 'same', activ)
            # x = conv2d(x, ch, ks, 1, 'same', activ)
            # x = conv2d(x, ch_in, ks, st, 'same', activ)

            # x = conv2d(x, ch, ks, 1, 'same', activ)
            # x = conv2d(x, ch*2, ks, st, 'same', activ)
            # x = conv2d(x, ch*2, ks, 1, 'same', activ)
            # x = conv2d(x, ch, ks, 1, 'same', activ)
            # x = conv2d(x, ch, ks, 1, 'same', activ)
            # x = conv2d(x, ch//2, ks, 1, 'same', activ)
            # x = conv2d(x, ch//2, ks, 1, 'same', activ)
            return x

class decoder:
    def __init__(self):
        pass

    def build(self, inputs, ch_out, ch=64):
        activ = tf.nn.relu
        ks = 4
        st = 2

        with tf.variable_scope("decoder", reuse=False):
            x = inputs
            for i in range(2):
                x = deconv2d(x, ch_out, ks, 1, 'same', activ)
                x = deconv2d(x, ch_out, ks, st, 'same', activ)
                x = deconv2d(x, ch_out, ks, 1, 'same', activ)
                x = deconv2d(x, ch_out, ks, st, 'same', activ)

            # x = deconv2d(x, ch//2, ks, 1, 'same', activ)
            # x = deconv2d(x, ch//2, ks, 1, 'same', activ)
            # x = deconv2d(x, ch, ks, 1, 'same', activ)
            # x = deconv2d(x, ch, ks, 1, 'same', activ)
            # x = deconv2d(x, ch*2, ks, st, 'same', activ)
            # x = deconv2d(x, ch*2, ks, 1, 'same', activ)
            # x = deconv2d(x, ch_out, ks, st, 'same', activ)
            # x = deconv2d(x, ch_out, ks, 1, 'same', activ)
            return x

class neuralnet:
    def __init__(self):
        pass

    def build(self, inputs, labels, ch):
        x = tf.layers.dense(inputs, units=ch, activation=tf.nn.softmax)
        x = tf.layers.dense(inputs, units=ch*2, activation=tf.nn.softmax)
        x = tf.layers.dense(inputs, units=ch*4, activation=tf.nn.softmax)
        x = tf.layers.dense(inputs, units=ch*2, activation=tf.nn.softmax)
        x = tf.layers.dense(inputs, units=ch, activation=tf.nn.softmax)
        x = tf.layers.dense(inputs, units=10, activation=tf.nn.softmax)
        # x = tf.layers.dense(x, units=10, activation=self.activ)
        return x

class Network:
    def __init__(self,
                 weight_initializer,
                 activation,
                 class_num,
                 patch_size,
                 patch_num,
                 ):
        self.weight_init = weight_initializer
        self.activ = activation
        self.cn = class_num
        self.ps = patch_size
        self.pn = patch_num

    def conv_3d(self, x, ch, ks, padding, activ, st = (1,1,1)):
        return tf.layers.conv3d(inputs=x,
                                filters=ch,
                                kernel_size=ks,
                                padding=padding,
                                activation=activ,
                                strides=st,
                                kernel_initializer=self.weight_init,
                                bias_initializer=self.weight_init)

    def deconv_3d(self, x, ch, ks, padding, activ, st = (1,1,1)):
        return tf.layers.conv3d_transpose(inputs=x,
                                          filters=ch,
                                          kernel_size=ks,
                                          padding=padding,
                                          activation=activ,
                                          strides=st,
                                          kernel_initializer=self.weight_init,
                                          bias_initializer=self.weight_init)

    def maxpool_3d(self, x, ps, st):
        return tf.layers.max_pooling3d(inputs=x, pool_size=ps, strides=st)

    def CNN_simple(self, x, ch = 32, scope = "CNN", reuse = False):
        k3 = [3,3,3]
        k4 = [4,4,4]
        k5 = [5,5,5]
        k7 = [7,7,7]
        # kernel_pool = [2,2,2]
        kernel_pool = [3,3,3]
        with tf.variable_scope(scope, reuse=reuse):
            x = batch_norm(x)
            x = self.conv_3d(x, ch, k5, 'same', self.activ)
            x = self.conv_3d(x, ch, k5, 'same', self.activ)
            x = self.maxpool_3d(x, kernel_pool, st=2)

            ch *= 2
            x = self.conv_3d(x, ch, k5, 'same', self.activ)
            x = self.conv_3d(x, ch, k5, 'same', self.activ)
            x = self.maxpool_3d(x, kernel_pool, st=2)

            ch *= 2
            x = self.conv_3d(x, ch, k5, 'same', self.activ)
            x = self.conv_3d(x, ch, k5, 'same', self.activ)
            x = self.maxpool_3d(x, kernel_pool, st=2)

            ch *= 2
            x = self.conv_3d(x, ch, k5, 'same', self.activ)
            x = self.conv_3d(x, ch, k5, 'same', self.activ)
            x = self.maxpool_3d(x, kernel_pool, st=2)
            return x

##################################################################################
# Convolutional Neural Network Model
##################################################################################
class Simple(Network):
    def model(self, images):
        is_print = False
        if is_print:
            print('build neural network')
            print(images.shape)

        channel = 32
        CNN = self.CNN_simple
        split_form = [self.ps for _ in range(self.pn)]
        with tf.variable_scope("Model"):
            # lh, rh = tf.split(images, split_form, 1)
            split_array = tf.split(images, split_form, 1)
            cnn_features = []
            for i, array in enumerate(split_array):
                array = CNN(array, ch=channel, scope="CNN"+str(i), reuse=False)
                array = tf.layers.flatten(array)
                cnn_features.append(array)
            # CNN = self.CNN_deep_layer

            # channel = 40
            # lh = CNN(lh, ch = channel, scope= "LCNN", reuse=False)
            # rh = CNN(rh, ch = channel, scope= "RCNN", reuse=False)
            with tf.variable_scope("FCN"):
                # lh = tf.layers.flatten(lh)
                # rh = tf.layers.flatten(rh)
                # x = tf.concat([lh, rh], -1)
                x = tf.concat(cnn_features, -1)
                x = tf.layers.dense(x, units=1024, activation=self.activ)
                x = tf.layers.dense(x, units=512, activation=self.activ)
                x = tf.layers.dense(x, units=self.cn, activation=tf.nn.softmax)
                # x = tf.layers.dense(x, units=self.cn, activation=tf.nn.sigmoid)
                y = x
        return y