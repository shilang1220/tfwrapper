#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-20 上午10:56
# @Author  : Guoliang PU
# @File    : multilayer_with_tensorflow.py
# @Software: tfwrapper

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def conv_model(input_data):
    # 第一层
    CONV1_FEATURES = 25
    MAX_POOL_SIZE1 = 2

    conv1_weight = tf.Variable(tf.truncated_normal([3, 3, CONV1_FEATURES], stddev=0.1, dtype=tf.float32))
    conv1_bias = tf.Variable(tf.zeros([CONV1_FEATURES], dtype=tf.float32))

    conv1 = tf.nn.conv2d(input_data, conv1_weight, strides=[1, 1, 1, 1], padding='SAME')
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_bias))
    max_pool1 = tf.nn.max_pool(relu1, ksize=[1, MAX_POOL_SIZE1, MAX_POOL_SIZE1, 1],
                               strides=[1, MAX_POOL_SIZE1, MAX_POOL_SIZE1, 1], padding='SAME')

    # 第二层
    CONV2_FEATURES = 50
    MAX_POOL_SIZE2 = 2
    conv2_weight = tf.Variable(tf.truncated_normal([3, 3, CONV2_FEATURES], stddev=0.1, dtype=tf.float32))
    conv2_bias = tf.Variable(tf.zeros([CONV2_FEATURES], dtype=tf.float32))

    conv2 = tf.nn.conv2d(max_pool1, conv2_weight, strides=[1, 1, 1, 1], padding='SAME')
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_bias))
    max_pool2 = tf.nn.max_pool(relu2, ksize=[1, MAX_POOL_SIZE2, MAX_POOL_SIZE2, 1],
                               strides=[1, MAX_POOL_SIZE2, MAX_POOL_SIZE2, 1], padding='SAME')

    # 全连接层1
    FULL_CONNECTED_SIZE1 = 100
    result_width = IMAGE_WIDTH // (MAX_POOL_SIZE1 * MAX_POOL_SIZE2)
    result_height = IMAGE_HEIGHT // (MAX_POOL_SIZE1 * MAX_POOL_SIZE2)
    full_connected_input_size = result_height * result_width

    full1_weight = tf.Variable(
        tf.truncated_normal([full_connected_input_size, FULL_CONNECTED_SIZE1], stddev=0.1, dtype=tf.float32))
    full1_bias = tf.Variable(tf.truncated_normal([FULL_CONNECTED_SIZE1], stddev=0.1, dtype=tf.float32))

    final_conv_shape = max_pool2.get_shape().as_list()
    final_shape = final_conv_shape[1] * final_conv_shape[2] * final_conv_shape[3]
    flat_output = tf.reshape(max_pool2, [final_conv_shape[0], final_shape])

    full_connected1 = tf.nn.relu(tf.add(tf.matmul(flat_output, full1_weight), full1_bias))

    # 全连接层2
    full2_weight = tf.Variable(tf.truncated_normal([FULL_CONNECTED_SIZE1, NUM_CLASSES], stddev=0.1, dtype=tf.float32))
    full2_bias = tf.Variable(tf.truncated_normal([NUM_CLASSES], stddev=0.1, dtype=tf.float32))

    full_connected2 = tf.add(tf.matmul(full_connected1, full2_weight), full2_bias)

    return(full_connected2)


sess = tf.Session()

#1. 定义数据
(train_x,train_y),(test_x,test_y) = tf.keras.datasets.mnist.load_data(path = 'mnist.npz')

print('Sample input x shape is:',train_x.shape)
print('Sample input y shape is:',train_x.shape)

#2. 定义参数和超参数
NUM_CLASSES = 10
IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28
IMAGE_CHANNELS = 1

NUM_SAMPLES = 60000
BATCH_SIZE = 32
LEARNING_RATE = 0.01
EPOCHS = 100

#3. 定义计算图, train_step = optim.minimize(loss)
input_shape = [BATCH_SIZE,IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_CHANNELS]
x_input = tf.placeholder(tf.float32,shape = input_shape)
y_target = tf.placeholder(tf.float32,shape = (BATCH_SIZE))

model_output = conv_model(x_input)

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(model_output,y_target))

optim = tf.train.MomentumOptimizer(LEARNING_RATE,0.9)

train_step = optim.minimize(loss)

#4. 初始化权重
init = tf.global_variables_initializer()
sess.run(init)

#5. 运行计算图，run(train_step)
for i in range(EPOCHS): #轮数
    #随机生成小批量
    rand_index = np.random.choice(len(train_x), size=BATCH_SIZE)
    rand_x = train_x[rand_index]
    rand_x = np.expand_dims(rand_x, 3)
    rand_y = train_y[rand_index]

    train_dict = {x_input: rand_x, y_target: rand_y}

    #更新权重
    sess.run(train_step, feed_dict=train_dict)


#6. 评估训练结果
eval_input_shape = [BATCH_SIZE,IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_CHANNELS]
eval_x_input = tf.placeholder(tf.float32,shape = input_shape)
eval_y_target = tf.placeholder(tf.float32,shape = (BATCH_SIZE))