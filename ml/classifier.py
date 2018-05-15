#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-15 下午5:32
# @Author  : Guoliang PU
# @File    : classifier.py
# @Software: tfwrapper

from keras.layers import Dense
from keras.models import Sequential
from keras.datasets import mnist

(x_train,y_train),(x_test,y_test) = mnist.load_data()
print('Origin train data shape',x_train.shape(),y_train.shape())
x_train = x_train / 255
x_test = x_test /255

