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
    with tf.name_scope('ConvLayer1')
        CONV1_FEATURES = 25
        MAX_POOL_SIZE1 = 2

        with tf.name_scope('Conv1'):
            conv1_weight = tf.Variable(tf.truncated_normal([3, 3, CONV1_FEATURES], stddev=0.1, dtype=tf.float32))
            conv1_bias = tf.Variable(tf.zeros([CONV1_FEATURES], dtype=tf.float32))

            conv1 = tf.nn.conv2d(input_data, conv1_weight, strides=[1, 1, 1, 1], padding='SAME')
        with tf.name_scope('Relu1'):
            relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_bias))
        with tf.name_scope('Maxpool1'):
            max_pool1 = tf.nn.max_pool(relu1, ksize=[1, MAX_POOL_SIZE1, MAX_POOL_SIZE1, 1],
                                   strides=[1, MAX_POOL_SIZE1, MAX_POOL_SIZE1, 1], padding='SAME')

    # 第二层
    with tf.name_scope('ConvLayer2')
        CONV2_FEATURES = 50
        MAX_POOL_SIZE2 = 2

        with tf.name_scope('Conv2'):
            conv2_weight = tf.Variable(tf.truncated_normal([3, 3, CONV2_FEATURES], stddev=0.1, dtype=tf.float32))
            conv2_bias = tf.Variable(tf.zeros([CONV2_FEATURES], dtype=tf.float32))
            conv2 = tf.nn.conv2d(max_pool1, conv2_weight, strides=[1, 1, 1, 1], padding='SAME')
        with tf.name_scope('Relu2'):
            relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_bias))
        with tf.name_scope('Maxpool2'):
            max_pool2 = tf.nn.max_pool(relu2, ksize=[1, MAX_POOL_SIZE2, MAX_POOL_SIZE2, 1],
                               strides=[1, MAX_POOL_SIZE2, MAX_POOL_SIZE2, 1], padding='SAME')

    # 全连接层1
    with tf.name_scope('FullCon1'):
        FULL_CONNECTED_SIZE1 = 100
        result_width = IMAGE_WIDTH // (MAX_POOL_SIZE1 * MAX_POOL_SIZE2)
        result_height = IMAGE_HEIGHT // (MAX_POOL_SIZE1 * MAX_POOL_SIZE2)
        full_connected_input_size = result_height * result_width

        full1_weight = tf.Variable(tf.truncated_normal([full_connected_input_size, FULL_CONNECTED_SIZE1], stddev=0.1, dtype=tf.float32))
        full1_bias = tf.Variable(tf.truncated_normal([FULL_CONNECTED_SIZE1], stddev=0.1, dtype=tf.float32))
        final_conv_shape = max_pool2.get_shape().as_list()
        final_shape = final_conv_shape[1] * final_conv_shape[2] * final_conv_shape[3]
        flat_output = tf.reshape(max_pool2, [final_conv_shape[0], final_shape])

        full_connected1 = tf.nn.relu(tf.add(tf.matmul(flat_output, full1_weight), full1_bias))

    # 全连接层2
    with tf.name_scope('FullCon2'):
        full2_weight = tf.Variable(tf.truncated_normal([FULL_CONNECTED_SIZE1, NUM_CLASSES], stddev=0.1, dtype=tf.float32))
        full2_bias = tf.Variable(tf.truncated_normal([NUM_CLASSES], stddev=0.1, dtype=tf.float32))

        full_connected2 = tf.add(tf.matmul(full_connected1, full2_weight), full2_bias)

    return(full_connected2)

#生成训练小批量数据集
def train_batch_generator(train_x,train_y,batch_size = 64):
    rand_index = np.random.choice(len(train_x), size=batch_size)
    batch_x = train_x[rand_index]
    batch_x = np.expand_dims(batch_x, 3)
    batch_y = train_y[rand_index]
    return batch_x,batch_y

#生成测试小批量数据集
def eval_batch_generator(eval_x,eval_y,batch_size = 64):
    rand_index = np.random.choice(len(eval_x), size=batch_size)
    eval_batch_x = eval_x[rand_index]
    eval_batch_x = np.expand_dims(eval_batch_x, 3)
    eval_batch_y = eval_y[rand_index]
    return eval_batch_x, eval_batch_y


#生成预测小批量数据集
def predict_batch_generator(predict_x, batch_size=64):
    rand_index = np.random.choice(len(predict_x), size=batch_size)
    predict_batch_x = predict_x[rand_index]
    predict_batch_x = np.expand_dims(predict_batch_x, 3)
    return predict_batch_x





# main function
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
BATCH_SIZE = 64
LEARNING_RATE = 0.01
EPOCHS = 100

#3. 定义计算图, train_step = optim.minimize(loss)
input_shape = [BATCH_SIZE,IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_CHANNELS]
x_input = tf.placeholder(tf.float32,shape = input_shape)
y_target = tf.placeholder(tf.float32,shape = (BATCH_SIZE))  #列向量，值表示类别

eval_x = tf.placeholder(tf.float32,shape = input_shape)
eval_y = tf.placeholder(tf.float32,shape = (BATCH_SIZE))

# 定义训练模型
train_model_output = conv_model(x_input)
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(train_model_output,y_target))
optim = tf.train.MomentumOptimizer(LEARNING_RATE,0.9)
train_step = optim.minimize(loss)

# 定义评估模型
eval_model_output = conv_model(eval_x)
eval_model_predict = tf.nn.softmax(eval_model_output)

#4. 初始化权重
init = tf.global_variables_initializer()

# 5. 运行计算图，run(train_step)
with tf.Session() as sess:
    #定义summary和saver
    tf.summary.scalar('Loss',loss)
    mearge_pos = tf.summary.merge_all()

    summary = tf.summary.FileWriter('logs')
    summary.add_graph(sess.graph)

    saver = tf.train.Saver()

    sess.run(init)

    for i in range(EPOCHS): #轮数

        #生成小批量
        batch_x,batch_y = train_batch_generator(x_input,y_target,batch_size=128)
        #定义注入变量
        train_dict = {x_input: batch_x, y_target: batch_y}
        #更新权重
        sess.run(train_step, feed_dict=train_dict)

        #每隔10轮后，评估一次模型
        if i % 10 == 0:
            eval_batch_x, eval_batch_y = eval_batch_generator(test_x, test_y, batch_size=128)
            all_summary = sess.run(mearge_pos,feed_dict = {x_input:eval_batch_x ,y_target:eval_batch_y})
            summary.add_summary(all_summary, global_step = i)

#7. 保存参数
    summary.close()
    saver.save(sess,'save/model.ckpt')


#6. 评估训练结果

eval_input_shape = [BATCH_SIZE,IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_CHANNELS]
eval_x_input = tf.placeholder(tf.float32,shape = input_shape)
eval_y_target = tf.placeholder(tf.float32,shape = (BATCH_SIZE))


