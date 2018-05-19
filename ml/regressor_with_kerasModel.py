#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-15 下午4:54
# @Author  : Guoliang PU
# @File    : regressor_with_kerasSquential.py
# @Software: tfwrapper

import numpy as np
np.random.seed(1337)

from keras.models import Model
from keras.layers import Dense
from keras.layers import Activation
import matplotlib.pyplot as plt

x = np.linspace(0,2*np.pi,200)
np.random.shuffle(x)
y = np.sin(x) + np.random.normal(0,0.05,(200,))

plt.scatter(x,y)
plt.show()

x_train,y_train = x[:160],y[:160]
x_test,y_test = x[160:],y[160:]



model = Sequential()

model.add(Dense(output_dim = 1,input_dim = 1,activation='tanh'))
model.add(Dense(output_dim = 1,activation='tanh'))
model.add(Dense(output_dim = 1,activation='tanh'))
model.add(Dense(output_dim = 1,activation='tanh'))
model.add(Dense(output_dim = 1,activation='tanh'))
#model.add(Activation('sig'))
model.compile(loss='mse',optimizer='sgd')

print('Training....')
for step in range (300):
    cost = model.train_on_batch(x_train,y_train)
    if step % 100 == 0:
        print('train cost:',cost)


print('Test....')
cost = model.evaluate(x_test,y_test,batch_size = 40)
print('Test cost:', cost)

W,b= model.layers[0].get_weights()
print('Weight = ',W,'\nbias=',b)

y_pred = model.predict(x_test)
plt.scatter(x_test,y_test)
#plt.plot(x_test,y_pred)
plt.show()