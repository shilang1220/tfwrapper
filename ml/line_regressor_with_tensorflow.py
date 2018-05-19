#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-19 上午10:36
# @Author  : Guoliang PU
# @File    : line_regressor_with_tensorflow.py
# @Software: tfwrapper


import numpy as np
import tensorflow as tf
from sklearn import datasets

#1. 定义数据
boston = datasets.load_boston()
x_vals = boston.data
y_vals = boston.target

feature_nums = x_vals.shape[1]

LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHES = 1000

#2.定义计算图
#从placeholder开始，到损失函数loss结束
x = tf.placeholder(tf.float32,shape=[None,feature_nums])
y = tf.placeholder(tf.float32,shape= [None,1])

w = tf.Variable(tf.random_normal(shape = [feature_nums,1]))
b = tf.Variable(tf.random_normal(shape = [1,1]))

y_=tf.add(tf.matmul(x,w),b)

loss = tf.reduce_mean(tf.square(y - y_))

#3.初始化权重参数

sess = tf.Session()
init_op = tf.global_variables_initializer()
sess.run(init_op)
print('权重初值为:')
print('W is:',sess.run(w))
print('b is:',sess.run(b))

#4. 定义优化算法
opt = tf.train.GradientDescentOptimizer(0.001)
train_step = opt.minimize(loss)

#5. 根据优化算法，更新权重参数100轮
for i in range(EPOCHES):
    sess.run(train_step,feed_dict={x:x_vals,y:y_vals.reshape([y_vals.shape[0],1])})
    print('第%d轮权重初值为:' % i)
    print('W is:', sess.run(w))
    print('b is:', sess.run(b))


