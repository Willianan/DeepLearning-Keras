# -*- coding:utf-8 -*-
"""
@Author：Charles Van
@E-mail:  williananjhon@hotmail.com
@Time：2019/4/8 9:27
@Project:DeepLearning-Keras
@Filename:load_data.py
"""


from keras.datasets import cifar10
from scipy.misc import toimage
import numpy as np
import matplotlib.pyplot as plt

# 导入数据
(x_train,y_train),(x_velication,y_velication) = cifar10.load_data()

for i in range(9):
	plt.subplot(331 + i)
	plt.imshow(toimage(x_train[i]))

plt.show()