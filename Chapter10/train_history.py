# -*- coding:utf-8 -*-
"""
@Author：Charles Van
@E-mail:  williananjhon@hotmail.com
@Time：2019/4/3 9:32
@Project:DeepLearning-Keras
@Filename:train_history.py
"""

from sklearn import datasets
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from matplotlib import pyplot as plt

# 导入数据
dataset = datasets.load_iris()
x = dataset.data
Y = dataset.target
# 将标签转换成分类编码
Y_labels = to_categorical(Y, num_classes=3)

# 设定随机种子
seed = 7
np.random.seed(seed)


# 构建模型函数
def create_model(optimizer='rmsprop', init='glorot_uniform'):
	# 构建模型
	model = Sequential()
	model.add(Dense(units=4, activation='relu', input_dim=4, kernel_initializer=init))
	model.add(Dense(units=6, activation='relu', kernel_initializer=init))
	model.add(Dense(units=3, activation='softmax', kernel_initializer=init))
	# 编译模型
	model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
	return model


# 构建模型
model = create_model()

history = model.fit(x, Y_labels, validation_split=0.2, epochs=200, batch_size=5, verbose=0)

# 评估模型
scores = model.evaluate(x, Y_labels, verbose=0)
print("%s : %.2f%%", (model.metrics_names[1], scores[1] * 100))

# history列表
print(history.history.keys())

# accuracy的历史数据
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('picture/accuracy.png')
plt.show()

# loss的历史数据
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('picture/loss.png')
plt.show()