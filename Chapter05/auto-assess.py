# -*- coding:utf-8 -*-
"""
@Author：Charles Van
@E-mail:  williananjhon@hotmail.com
@Time：2019/4/1 10:16
@Project:DeepLearning-Keras
@Filename:auto-assess.py
"""

from keras.models import Sequential
from keras.layers import Dense
import numpy as np


class AutoAssess:
	def loadDataSet(self, filename):
		return np.loadtxt(filename, delimiter=',')

	def AutoAssessModel(self, dataSet):
		# 设定随机数种子
		np.random.seed(7)
		# 分割输入x和输出Y
		x = dataSet[:, 0:8]
		Y = dataSet[:, 8]
		# 创建模型
		model = Sequential()
		model.add(Dense(12, input_dim=8, activation='relu'))
		model.add(Dense(8, activation='relu'))
		model.add(Dense(1, activation='sigmoid'))
		# 编译模型
		model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
		# 训练模型并自动评估模型
		model.fit(x=x, y=Y, epochs=150, batch_size=10, validation_split=0.2)


if __name__ == "__main__":
	test = AutoAssess()
	dataSet = test.loadDataSet("pima-indians-diabetes.csv")
	test.AutoAssessModel(dataSet)
