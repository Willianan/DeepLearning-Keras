# -*- coding:utf-8 -*-
"""
@Author：Charles Van
@E-mail:  williananjhon@hotmail.com
@Time：2019/4/7 11:21
@Project:DeepLearning-Keras
@Filename:zca_whitening.py
"""

# ZCA白化
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from keras import backend

backend.set_image_data_format('channels_first')

# 导入数据
(X_train,Y_train),(X_velication,Y_velication) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0],1,28,28).astype("float32")
X_velication = X_velication.reshape(X_velication.shape[0],1,28,28).astype("float32")

# ZCA白化
imgGen = ImageDataGenerator(zca_whitening=True)
imgGen.fit(X_train)

for X_batch,Y_batch in imgGen.flow(X_train,Y_train,batch_size=9):
	for i in range(9):
		plt.subplot(331 + i)
		plt.imshow(X_batch[i].reshape(28,28),cmap=plt.get_cmap('gray'))
	plt.show()
	break