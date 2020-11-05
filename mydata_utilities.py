import tensorflow as tf
import tensorflow.contrib.slim as slim

"""MNIST BiGAN architecture.

Generator (decoder), encoder and discriminator.

"""

learning_rate = 0.00001
batch_size = 4
latent_dim = [16, 16, 64]


# conv2d
def conv(inputs, out_channels, kernel_size=3, stride=1):
    '''
    inputs: tensor
    out_channels: output channels  int
    kernel_size: kernel size int
    stride: int
    return:tensor
    ...
    conv2d:
        input : [batch, height, width, channel]
        kernel : [height, width, in_channels, out_channels]
    '''
    # fixed edge of tensor
    if stride > 1:
        inputs = _fixed_padding(inputs, kernel_size)
    #
    inputs = slim.conv2d(inputs, out_channels, kernel_size, stride=stride,
                         padding=('SAME' if stride == 1 else 'VALID'))
    return inputs


@tf.contrib.framework.add_arg_scope
def _fixed_padding(inputs, kernel_size, mode='CONSTANT'):
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                    [pad_beg, pad_end], [0, 0]], mode=mode)
    return padded_inputs


def encoder(x_inp, is_training=False, getter=None, reuse=False):
    """ Encoder architecture in tensorflow

    Maps the data into the latent space

    Args:
        x_inp (tensor): input data for the encoder.
        reuse (bool): sharing variables or not

    Returns:
        (tensor): last activation layer of the encoder

    """
    batch_norm_params = {"decay": 0.9997000098228455,
                         "epsilon": 0.0010000000474974513,
                         "scale": True,
                         "center": True,
                         "is_training": is_training}
    with slim.arg_scope([slim.conv2d],
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params,
                        activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=.2)):
        with tf.variable_scope('encoder', reuse=reuse, custom_getter=getter):
            n = tf.reshape(x_inp, [-1, 256, 256, 3])
            n = conv(n, 64, 7)

            n = conv(n, 64)
            n = conv(n, 128, 1)
            n = tf.nn.max_pool(n, ksize=(1, 3, 3, 1), strides=(1, 2, 2, 1), padding='SAME')

            n = conv(n, 128)
            n = conv(n, 128, 1)
            n = tf.nn.max_pool(n, ksize=(1, 3, 3, 1), strides=(1, 2, 2, 1), padding='SAME')

            n = conv(n, 128)
            for i in range(2):
                route = n
                n = conv(n, 128, 1)
                n = conv(n, 256)
                n = conv(n, 128, 1)
                n += route
            n = tf.nn.max_pool(n, ksize=(1, 3, 3, 1), strides=(1, 2, 2, 1), padding='SAME')

            n = conv(n, 64)
            for i in range(3):
                route = n
                n = conv(n, 64, 1)
                n = conv(n, 128)
                n = conv(n, 64, 1)
                n += route
            n = tf.nn.max_pool(n, ksize=(1, 3, 3, 1), strides=(1, 2, 2, 1), padding='SAME')

            n = slim.conv2d(n, 64, 3,
                            stride=1, normalizer_fn=None,
                            activation_fn=None,
                            biases_initializer=tf.zeros_initializer())
    return n  # 16*16*64


def decoder(z_inp, is_training=False, getter=None, reuse=False):
    """ Decoder architecture in tensorflow

    Generates data from the latent space

    Args:
        z_inp (tensor): variable in the latent space
        reuse (bool): sharing variables or not

    Returns:
        (tensor): last activation layer of the generator

    """
    batch_norm_params = {"decay": 0.9997000098228455,
                         "epsilon": 0.0010000000474974513,
                         "scale": True,
                         "center": True,
                         "is_training": is_training}
    with slim.arg_scope([slim.conv2d],
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params,
                        biases_initializer=None,
                        activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=.2)):
        with tf.variable_scope('generator', reuse=reuse, custom_getter=getter):
            n = slim.conv2d(z_inp, 128, 3,
                            stride=1, normalizer_fn=None,
                            activation_fn=None,
                            biases_initializer=tf.zeros_initializer())
            n = conv(n, 64)
            # 开始深度残差网络
            route = n
            for i in range(3):
                nn = conv(n, 64, 1)
                nn = conv(nn, 128)
                nn = conv(nn, 64, 1)
                n += nn
            n += route
            n = slim.conv2d_transpose(n, 128, 3, stride=2, activation_fn=None)

            route = n
            for i in range(2):
                nn = conv(n, 128, 1)
                nn = conv(nn, 256)
                nn = conv(nn, 128, 1)
                n += nn
            n += route
            n = slim.conv2d_transpose(n, 128, 3, stride=2, activation_fn=None)

            n = conv(n, 128)
            n = conv(n, 256, 1)
            n = slim.conv2d_transpose(n, 256, 3, stride=2, activation_fn=None)

            n = conv(n, 256)
            n = conv(n, 128, 1)
            n = slim.conv2d_transpose(n, 128, 3, stride=2, activation_fn=None)

            n = conv(n, 64)
            n = slim.conv2d(n, 3, 1,
                            stride=1, normalizer_fn=None,
                            activation_fn=None,
                            biases_initializer=tf.zeros_initializer())
            n = tf.nn.tanh(n)
    return n


def discriminator(z_inp, x_inp, is_training=False, getter=None, reuse=False):
    """ Discriminator architecture in tensorflow

    Discriminates between pairs (E(x), x) and (z, G(z))

    Args:
        z_inp (tensor): variable in the latent space
        x_inp (tensor): input data for the encoder.
        reuse (bool): sharing variables or not

    Returns:
        logits (tensor): last activation layer of the discriminator
        intermediate_layer (tensor): intermediate layer for feature matching

    """
    batch_norm_params = {"decay": 0.9997000098228455,
                         "epsilon": 0.0010000000474974513,
                         "scale": True,
                         "center": True,
                         "is_training": is_training}
    with slim.arg_scope([slim.conv2d],
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params,
                        biases_initializer=None,
                        activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=.2)):
        with tf.variable_scope('discriminator', reuse=reuse, custom_getter=getter):
            x_inp = tf.reshape(x_inp, [-1, 256, 256, 3])
            z_inp = tf.reshape(z_inp, [-1, 16, 16, 64])

            n = conv(x_inp, 64, 7)
            n = conv(n, 64)
            n = conv(n, 128, 1)
            n = tf.nn.max_pool(n, ksize=(1, 3, 3, 1), strides=(1, 2, 2, 1), padding='SAME')

            n = conv(n, 128)
            n = conv(n, 128, 1)
            n = tf.nn.max_pool(n, ksize=(1, 3, 3, 1), strides=(1, 2, 2, 1), padding='SAME')

            n = conv(n, 128)
            for i in range(2):
                route = n
                n = conv(n, 128, 1)
                n = conv(n, 256)
                n = conv(n, 128, 1)
                n += route
            n = tf.nn.max_pool(n, ksize=(1, 3, 3, 1), strides=(1, 2, 2, 1), padding='SAME')

            n = conv(n, 64)
            for i in range(3):
                route = n
                n = conv(n, 64, 1)
                n = conv(n, 128)
                n = conv(n, 64, 1)
                n += route
            n = tf.nn.max_pool(n, ksize=(1, 3, 3, 1), strides=(1, 2, 2, 1), padding='SAME')
            n = tf.concat([n,z_inp], axis=-1)

            n = conv(n, 64, stride=2)
            n = conv(n, 128, stride=2)
            n = conv(n, 256, stride=2)
            n = slim.conv2d(n, 512, 1,
                            stride=2, normalizer_fn=None,
                            activation_fn=None,
                            biases_initializer=tf.zeros_initializer())
            intermediate_layer = tf.reshape(n, [batch_size, -1])
            logits = tf.layers.dense(n, 1, name='fc')

    return logits, intermediate_layer
