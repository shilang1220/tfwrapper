#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-19 上午10:36
# @Author  : Guoliang PU
# @File    : line_regressor_with_tensorflow.py
# @Software: tfwrapper


import numpy as np
import tensorflow as tf
from sklearn import datasets

# 1. 定义数据
from tensorflow.python.summary.writer.writer import FileWriter

boston = datasets.load_boston()
x_vals = boston.data
y_vals = boston.target
y_vals = y_vals.reshape(y_vals.shape[0], 1)

feature_num = x_vals.shape[1]
sample_num = x_vals.shape[0]

print('Sample number is: %d' % sample_num)
print('Feature number is: %d' % feature_num)

LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHES = 1000

# 2.定义计算图
# 从placeholder开始，到损失函数loss结束
with tf.name_scope('input_x'):
    x = tf.placeholder(tf.float32, shape=[None, feature_num])
with tf.name_scope('input_y'):
    y = tf.placeholder(tf.float32, shape=[None, 1])

with tf.name_scope('layer'):
    with tf.name_scope('Weight'):
        w = tf.Variable(tf.random_normal(shape=[feature_num, 1]))
    with tf.name_scope('bias'):
        b = tf.Variable(tf.random_normal(shape=[1, 1]))
    with tf.name_scope('Wx_plus_b'):
        y_ = tf.add(tf.matmul(x, w), b)

# 3.定义损失函数
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.square(y - y_))

# 4.定义优化算法
with tf.name_scope('train'):
    opt = tf.train.GradientDescentOptimizer(0.001)
    train_step = opt.minimize(loss)

# 5.初始化权重参数
sess = tf.Session()
train_writer: FileWriter = tf.summary.FileWriter('logs/train', sess.graph)
validate_writer = tf.summary.FileWriter('logs/validate')

init_op = tf.global_variables_initializer()
sess.run(init_op)
print('权重初值为:')
print('W is:', sess.run(w))
print('b is:', sess.run(b))

# 6. 根据优化算法，更新权重参数
for i in range(EPOCHES):
    sess.run(train_step, feed_dict={x: x_vals, y: y_vals.reshape([y_vals.shape[0], 1])})
    print('第%d轮权重初值为:' % i)
    print('W is:', sess.run(w))
    print('b is:', sess.run(b))

# 7. 生成日志
train_writer.close()
validate_writer.close

# 8. 保存模型
saver = tf.train.Saver()
saver.save(sess, 'save/line_regressor.ckpt')
