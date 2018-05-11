#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-11 下午12:21
# @Author  : Guoliang PU
# @File    : model.py
# @Software: tfwrapper

from core import network


class Model(network):

    def __init__(self):
        pass

    def fit(self):
        '''
        基于全训练数据集进行训练拟合
        :return:
        '''
        pass

    def train_on_batch(self):
        '''
        小批量训练拟合
        :return:
        '''
        pass

    def _train_batch_generator(self):
        pass

    def evaluate(self):
        pass

    def test_on_batch(self):
        pass

    def _test_batch_generator(self):
        pass

    def predict(self):
        pass

    def predict_on_batch(self):
        pass

    def _predict_batch_generator(self):
        pass
