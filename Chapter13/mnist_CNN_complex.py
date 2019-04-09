# -*- coding:utf-8 -*-
"""
@Author：Charles Van
@E-mail:  williananjhon@hotmail.com
@Time：2019/4/7 9:51
@Project:DeepLearning-Keras
@Filename:mnist_CNN_complex.py
"""
"""
#   复杂卷积神经网络的网络拓扑结构：
(1)、卷积层，具有30个特征图，感受野大小为5x5
(2)、采样因子（pool_size）为2x2的池化层
(3)、卷积层，具有15个特征图，感受野大小为3x3
(4)、采样因子（pool_size）为2x2的池化层
(5)、Dropout概率为20%的Dropout层
(6)、Flatten层
(7)、具有128个神经元和ReLU激活函数的全连接层
(8)、具有50个神经元和ReLU激活函数的全连接层
(9)、输出层
"""

from keras.datasets import mnist
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend
backend.set_image_data_format('channels_first')

# 设定随机数种子
seed = 7
np.random.seed(seed)

# 导入mnist数据集
(X_train,Y_train),(X_validation,Y_validation) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0],1,28,28).astype('float32')
X_validation = X_validation.reshape(X_validation.shape[0],1,28,28).astype('float32')

# 格式化数据到0~1
X_train = X_train / 255
X_validation = X_validation / 255

# 进行one-hot编码
Y_train = np_utils.to_categorical(Y_train)
Y_validation = np_utils.to_categorical(Y_validation)

# 创建模型
def create_model():
	model = Sequential()
	model.add(Conv2D(30,(5,5),input_shape=(1,28,28),activation='relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Conv2D(15,(3,3),activation='relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(units=128,activation='relu'))
	model.add(Dense(units=50,activation='relu'))
	model.add(Dense(units=10,activation='softmax'))

	# 编译模型
	model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
	return model

model = create_model()
model.fit(X_train,Y_train,epochs=10,batch_size=200,verbose=2)
scores = model.evaluate(X_validation,Y_validation,verbose=0)
print("CNN_Large : %.2f%%" % (scores[1] * 100))