import numpy as np
import tensorflow as tf
##################################################################################
# Custom Operation
##################################################################################
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

    def resblock(self, input, ch_in, ch_out, ks):
        x = self.conv_3d(input, ch_out, 1, 'same', self.activ)
        x = self.conv_3d(x, ch_out, ks, 'same', self.activ)
        if ch_in != ch_out:
            input = self.conv_3d(input, ch_out, ks, 'same', self.activ)
        return self.conv_3d(x, ch_out, ks, 'same', self.activ) + input


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

class Attention(Network):
    def CNN_attention_default(self, input, ch = 16, scope = "resAttention", reuse = False):
        k3 = [3, 3, 3]
        k4 = [4, 4, 4]
        k5 = [5, 5, 5]
        k7 = [7, 7, 7]
        t = 1
        p = 1
        r = 1
        with tf.variable_scope(scope):
            # residual blocks(TODO: change this function)
            with tf.variable_scope("first_residual_blocks"):
                for i in range(t):
                    x = self.resblock(input, 1, ch, ks=3)

            with tf.variable_scope("trunk_branch"):
                output_trunk = x
                for i in range(t):
                    output_trunk = self.resblock(x, ch, ch, ks=3)

            with tf.variable_scope("soft_mask_branch"):
                with tf.variable_scope("down_sampling_1"):
                    # max pooling
                    filter_ = 3
                    output_soft_mask = self.maxpool_3d(input, ps=2, st=2)
                    for i in range(r):
                        output_soft_mask = self.resblock(output_soft_mask, ch, ch, ks=3)

                with tf.variable_scope("skip_connection"):
                    output_skip_connection = self.resblock(output_soft_mask, ch, ch, ks=3)

                with tf.variable_scope("down_sampling_2"):
                    # max pooling
                    filter_pool = [2,2,2]
                    output_soft_mask = self.maxpool_3d(output_soft_mask, ps=2, st=2)
                    for i in range(r):
                        output_soft_mask = self.resblock(output_soft_mask, ch, ch, ks=3)

                with tf.variable_scope("up_sampling_1"):
                    for i in range(r):
                        output_soft_mask = self.resblock(output_soft_mask, ch, ch, ks=3)

                    # interpolation
                    output_soft_mask = self.deconv_3d(output_soft_mask, ch, k3, 'same', self.activ, st=2)

                    # output_soft_mask = UpSampling3D(size=(2,2,2))(output_soft_mask)
                    # output_soft_mask = UpSampling2D([2, 2])(output_soft_mask)

                # add skip connection
                output_soft_mask += output_skip_connection

                with tf.variable_scope("up_sampling_2"):
                    for i in range(r):
                        output_soft_mask = self.resblock(output_soft_mask, ch, ch, ks=3)
                    # interpolation
                    output_soft_mask = self.deconv_3d(output_soft_mask, ch, k3, 'same', self.activ, st=2)
                    # output_soft_mask = UpSampling3D(size=(2,2,2))(output_soft_mask)

                with tf.variable_scope("output"):
                    output_soft_mask = self.conv_3d(output_soft_mask, ch, k3, 'same', self.activ)
                    output_soft_mask = self.conv_3d(output_soft_mask, ch, k3, 'same', self.activ)
                    # sigmoid
                    output_soft_mask = tf.nn.sigmoid(output_soft_mask)

            with tf.variable_scope("attention"):
                output = (1 + output_soft_mask) * output_trunk

            with tf.variable_scope("last_residual_blocks"):
                for i in range(t):
                    output = self.resblock(output_soft_mask, ch, ch, ks=3)

            return output

    def attention(self, x, ch_in, ch = 16, depth = 1, scope = 'attention'):
        k3, k4, k5, k7 = 3,4,5,7
        r,p,t = 2,2,2
        skip = []
        input = x
        with tf.variable_scope(scope):
            with tf.variable_scope(scope+'_encode'):
                x = self.conv_3d(x, ch, k7, 'same', self.activ)
                for i in range(depth):
                    for j in range(r):
                        x = self.conv_3d(x, ch, 3, 'same', self.activ)
                    skip.append(x)
                    x = self.maxpool_3d(x, ps=2, st=2)

            for i in range(p):
                x = self.conv_3d(x, ch, k3, 'same', self.activ)

            with tf.variable_scope(scope+'_decode'):
                for i in range(depth):
                    x = self.deconv_3d(x, ch, k3, 'same', self.activ, st=2)
                    x = x + skip[depth - 1 - i]
                    for j in range(r):
                        x = self.conv_3d(x, ch, 3, 'same', self.activ)
                    # extract probability map by sigmoid activation function
                    x = self.conv_3d(x, ch_in, 3, 'same', tf.nn.sigmoid)
                    soft_mask = x

            for i in range(t):
                out = x = self.conv_3d(input, ch, k3, 'same', self.activ)
                out = x = self.conv_3d(out, ch_in, k3, 'same', self.activ)
                out = (1+soft_mask)*out
            return out

    def model(self, images):
        is_print = False
        if is_print:
            print('build neural network')
            print(images.shape)

        ch = 32
        CNN = self.CNN_attention_res
        split_form = [self.ps for _ in range(self.pn)]
        with tf.variable_scope("Model"):
            # lh, rh = tf.split(images, split_form, 1)
            split_array = tf.split(images, split_form, 1)
            cnn_features = []
            for i, x in enumerate(split_array):
                with tf.variable_scope("patch"+str(i)):
                    x = self.attention(x, 1, ch, depth = 2, scope='attent1')
                    x = self.conv_3d(x, ch, 3, 'same', self.activ, st=1)
                    x = self.conv_3d(x, ch, 3, 'same', self.activ, st=1)
                    x = self.maxpool_3d(x, ps=2, st=2)

                    x = self.attention(x, ch, ch*2, depth = 1, scope='attent2')
                    x = self.conv_3d(x, ch*2, 3, 'same', self.activ, st=1)
                    x = self.conv_3d(x, ch*2, 3, 'same', self.activ, st=1)
                    x = self.maxpool_3d(x, ps=2, st=2)

                    ch *= 2
                    x = self.attention(x, ch, ch * 2, depth=1, scope='attent3')
                    x = self.conv_3d(x, ch * 2, 3, 'same', self.activ, st=1)
                    x = self.conv_3d(x, ch * 2, 3, 'same', self.activ, st=1)
                    x = self.maxpool_3d(x, ps=2, st=2)

                    x = self.attention(x, ch, ch * 2, depth=1, scope='attent4')
                    x = self.conv_3d(x, ch * 2, 3, 'same', self.activ, st=1)
                    x = self.conv_3d(x, ch * 2, 3, 'same', self.activ, st=1)
                    x = self.maxpool_3d(x, ps=2, st=2)

                    # ch *= 2
                    # x = self.attention(x, ch, ch * 2, depth=1, scope='attent4')
                    # x = self.conv_3d(x, ch * 2, 3, 'same', self.activ, st=1)
                    # x = self.conv_3d(x, ch * 2, 3, 'same', self.activ, st=1)
                    # x = self.maxpool_3d(x, ps=2, st=2)

                    # x = CNN(x, ch=ch, scope="CNN"+str(i), reuse=False)
                    # x = self.maxpool_3d(x, ps=2, st=2)
                    # x = CNN(x, ch=ch*2, scope="CNN2"+str(i), reuse=False)
                    # x = self.maxpool_3d(x, ps=2, st=2)
                    # x = CNN(x, ch=ch*4, scope="CNN3"+str(i), reuse=False)
                    # x = self.maxpool_3d(x, ps=2, st=2)
                    # x = CNN(x, ch=ch*4, scope="CNN4"+str(i), reuse=False)
                    # x = self.maxpool_3d(x, ps=2, st=2)

                    # we need to reduce the image size..
                    # if i use residual attention module only one block, i can't reduce the image
                    # for i in range(2):
                    #     x = self.resblock(x, ch, ch*2, ks=3)
                    #     # x = self.conv_3d(x, ch, 3, 'same', self.activ)
                    #     x = self.maxpool_3d(x, ps=2, st=2)

                print(np.shape(x))
                x = tf.layers.flatten(x)
                cnn_features.append(x)
            # print(np.shape(cnn_features))
            with tf.variable_scope("FCN"):
                # x = tf.concat([lh, rh], -1)
                x = tf.concat(cnn_features, -1)
                x = tf.layers.dense(x, units=1024, activation=self.activ)
                x = tf.layers.dense(x, units=512, activation=self.activ)
                x = tf.layers.dense(x, units=self.cn, activation=tf.nn.softmax)
                # x = tf.layers.dense(x, units=self.cn, activation=tf.nn.sigmoid)
                y = x
        return y