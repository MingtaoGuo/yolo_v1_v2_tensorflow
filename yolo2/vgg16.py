import tensorflow as tf
import numpy as np

def conv(inputs, w, b, trainable=False):
    w = tf.Variable(initial_value=w, trainable=trainable)
    b = tf.Variable(initial_value=b, trainable=trainable)
    inputs = tf.nn.conv2d(inputs, w, [1, 1, 1, 1], "SAME")
    return tf.nn.bias_add(inputs, b)

def max_pooling(inputs):
    return tf.nn.max_pool(inputs, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")

def relu(inputs):
    return tf.nn.relu(inputs)

def vgg_fully_connected(name, inputs, init_W, init_b):
    with tf.variable_scope(name):
        inputs = tf.layers.flatten(inputs)
        W = tf.Variable(initial_value=init_W)
        b = tf.Variable(initial_value=init_b)
        inputs = tf.matmul(inputs, W)
        inputs = tf.nn.bias_add(inputs, b)
    return inputs

def conv_(name, inputs, nums_out, k_size, strides, padding="SAME"):
    nums_in = inputs.shape[-1]
    with tf.variable_scope(name):
        W = tf.get_variable("W", [k_size, k_size, nums_in, nums_out], initializer=tf.truncated_normal_initializer(stddev=0.02))
        b = tf.get_variable("b", [nums_out], initializer=tf.constant_initializer([0.]))
        inputs = tf.nn.conv2d(inputs, W, [1, strides, strides, 1], padding=padding)
        inputs = tf.nn.bias_add(inputs, b)
    return inputs

def fc(name, inputs, nums_out):
    nums_in = inputs.shape[-1]
    with tf.variable_scope(name):
        W = tf.get_variable("W", [nums_in, nums_out], initializer=tf.truncated_normal_initializer(stddev=0.02))
        b = tf.get_variable("b", [nums_out], initializer=tf.constant_initializer([0.]))
        inputs = tf.matmul(inputs, W)
    return tf.nn.bias_add(inputs, b)


def vgg16(inputs):
    inputs = tf.reverse(inputs, [-1]) - tf.constant([103.939, 116.779, 123.68])
    inputs /= 255.0
    para = np.load("./vgg_para/vgg16.npy", encoding="latin1").item()
    inputs = relu(conv(inputs, para["conv1_1"][0], para["conv1_1"][1]))
    inputs = relu(conv(inputs, para["conv1_2"][0], para["conv1_2"][1]))
    inputs = max_pooling(inputs)
    inputs = relu(conv(inputs, para["conv2_1"][0], para["conv2_1"][1]))
    inputs = relu(conv(inputs, para["conv2_2"][0], para["conv2_2"][1]))
    inputs = max_pooling(inputs)
    inputs = relu(conv(inputs, para["conv3_1"][0], para["conv3_1"][1], True))
    inputs = relu(conv(inputs, para["conv3_2"][0], para["conv3_2"][1], True))
    inputs = relu(conv(inputs, para["conv3_3"][0], para["conv3_3"][1], True))
    inputs = max_pooling(inputs)
    inputs = relu(conv(inputs, para["conv4_1"][0], para["conv4_1"][1], True))
    inputs = relu(conv(inputs, para["conv4_2"][0], para["conv4_2"][1], True))
    inputs = relu(conv(inputs, para["conv4_3"][0], para["conv4_3"][1], True))
    inputs = max_pooling(inputs)
    inputs = relu(conv(inputs, para["conv5_1"][0], para["conv5_1"][1], True))
    inputs = relu(conv(inputs, para["conv5_2"][0], para["conv5_2"][1], True))
    inputs = relu(conv(inputs, para["conv5_3"][0], para["conv5_3"][1], True))
    inputs = max_pooling(inputs)
    # inputs = tf.layers.flatten(inputs)
    # inputs = fc("fc", inputs, 512)
    # inputs = fc("logits", inputs, 13*13*9*5)
    # inputs = tf.reshape(inputs, [-1, 13, 13, 9, 5])
    return inputs

