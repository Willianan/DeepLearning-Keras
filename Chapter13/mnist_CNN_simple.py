# -*- coding:utf-8 -*-
"""
@Author：Charles Van
@E-mail:  williananjhon@hotmail.com
@Time：2019/4/7 9:51
@Project:DeepLearning-Keras
@Filename:mnist_CNN_simple.py
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


# 设定随机种子
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
	model.add(Conv2D(32,(3,5),input_shape=(1,28,28),activation='relu'))     # Conv2D卷积层
	model.add(MaxPooling2D(pool_size=(2,2)))                                # MaxPooling2D池化层
	model.add(Dropout(0.2))                                                 # Dropout正则化层
	model.add(Flatten())                                                    #Faltten层（多维数据转一维数据）
	model.add(Dense(units=128,activation='relu'))                           # 全连接层
	model.add(Dense(units=10,activation='softmax'))
	# 编译模型
	model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
	return model

model = create_model()
model.fit(X_train,Y_train,epochs=10,batch_size=200,verbose=2)         # berbose设置为2，仅输出每个epoch的最终结果

scores = model.evaluate(X_validation,Y_validation,verbose=0)
print("CNN_Small : %.2f%%" % (scores[1] * 100))
