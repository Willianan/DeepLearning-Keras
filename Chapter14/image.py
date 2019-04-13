# -*- coding:utf-8 -*-
"""
@Author：Charles Van
@E-mail:  williananjhon@hotmail.com
@Time：2019/4/7 11:02
@Project:DeepLearning-Keras
@Filename:image.py
"""

# 增强前的图像

from keras.datasets import mnist
import matplotlib.pyplot as plt

(X_train,X_valication),(Y_train,Y_valication) = mnist.load_data()

# 显示9张手写数字的图片
for i in range(9):
	plt.subplot(331 + i)
	plt.imshow(X_train[i],cmap=plt.get_cmap('gray'))

plt.show()