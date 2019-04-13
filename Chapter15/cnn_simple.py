# -*- coding:utf-8 -*-
"""
@Author：Charles Van
@E-mail:  williananjhon@hotmail.com
@Time：2019/4/8 9:35
@Project:DeepLearning-Keras
@Filename:cnn_simple.py
"""

"""
简单卷积神经网络：卷积层（2个），池化层，flatten层，全连接层
(1)、卷积层，具有32个特征图，感受野大小为3x3
(2)、Dropout层，Dropout概率为20%
(3)、卷积层，具有32个特征图，感受野大小为3x3
(4)、Dropout概率为20%的Dropout层
(5)、采样因子（pool_size）为2x2的池化层
(6)、Flatten层
(7)、具有512个神经元和ReLu激活函数的全连接层
(8)、Dropout概率为50%的Dropout层
(9)具有10个神经元的输出层，激活函数为softmax
"""

import numpy as np
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.optimizers import SGD
from keras.constraints import maxnorm
from keras.utils import np_utils
from keras import backend

backend.set_image_data_format('channels_first')

# 设定随机种子
seed = 7
np.random.seed(seed)

# 加载数据
(x_train, y_train), (x_validation, y_validation) = cifar10.load_data()

# 格式化数据到0~1
x_train = x_train.astype('float32')
x_validation = x_validation.astype('float32')
x_train = x_train / 255.0
x_validation = x_validation / 255.0

# 进行one-hot编码
y_train = np_utils.to_categorical(y_train)
y_validation = np_utils.to_categorical(y_validation)
num_classes = y_train.shape[1]


# 构建模型
def create_model(epochs=25):
	model = Sequential()
	model.add(Conv2D(32,(3,3),input_shape=(3,32,32),padding='same',activation='relu',kernel_constraint=maxnorm(3)))
	model.add(Dropout(0.2))
	model.add(Conv2D(32,(3,3),activation='relu',padding='same',kernel_constraint=maxnorm(3)))
	model.add(Dropout(0.2))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Flatten())
	model.add(Dense(units=512,activation='relu',kernel_constraint=maxnorm(3)))
	model.add(Dropout(0.5))
	model.add(Dense(units=10,activation='softmax'))
	lrate = 0.01
	decay = lrate / epochs
	sgd = SGD(lr=lrate,momentum=0.9,decay=decay,nesterov=False)
	model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
	return model


model = create_model(epochs=25)
model.fit(x=x_train,y=y_train,epochs=25,batch_size=32,verbose=2)
scores = model.evaluate(x_validation,y_validation,verbose=0)
print("Accuracy : %.2f%%" % (scores[1] * 100))
