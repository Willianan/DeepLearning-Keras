# -*- coding:utf-8 -*-
"""
@Author：Charles Van
@E-mail:  williananjhon@hotmail.com
@Time：2019/4/10 11:01
@Project:DeepLearning-Keras
@Filename:Simple_LSTM_CNN.py
"""

import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.preprocessing import sequence

seed = 7
top_words = 5000
max_words = 500
out_dimension = 32
batch_size = 128
epochs = 2
dropout_rate = 0.2


# 构建模型
def build_model():
	model = Sequential()
	model.add(Embedding(top_words, out_dimension, input_length=max_words))
	model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
	model.add(MaxPooling1D(pool_size=2))
	model.add(LSTM(units=100))
	model.add(Dense(units=1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	# 输出模型的概要信息
	model.summary()
	return model


if __name__ == '__main__':
	np.random.seed(seed=seed)
	# 导入数据
	(x_train, y_train), (x_validation, y_validation) = imdb.load_data(num_words=top_words)

	# 限定数据集的长度
	x_train = sequence.pad_sequences(x_train, maxlen=max_words)
	x_validation = sequence.pad_sequences(x_validation, maxlen=max_words)

	# 生产模型并训练模型
	model = build_model()
	model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=2)
	scores = model.evaluate(x_validation, y_validation, verbose=2)
	print('Accuracy: %.2f%%' % (scores[1] * 100))
