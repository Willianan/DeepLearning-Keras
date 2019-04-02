# -*- coding:utf-8 -*-
"""
@Author：Charles Van
@E-mail:  williananjhon@hotmail.com
@Time：2019/4/1 9:46
@Project:DeepLearning-Keras
@Filename:sequential_sample.py
"""

from keras.models import Sequential
from keras.layers import Dense
import numpy as np


class Test:
	def loadData(self, filename):
		dataSet = np.loadtxt(filename, delimiter=',')
		return dataSet

	def sequentialModel(self, dataSet):
		# 设定随机数种子
		np.random.seed(7)

		# 分割输入x和输出y
		x = dataSet[:, 0: 8]
		y = dataSet[:, 8]

		# 创建模型
		model = Sequential()
		model.add(Dense(12, input_dim=8, activation='relu'))
		model.add(Dense(8, activation='relu'))
		model.add(Dense(1, activation="sigmoid"))

		# 编译模型
		model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

		# 训练模型
		model.fit(x=x, y=y, epochs=150, batch_size=10)

		# 评估模型
		scores = model.evaluate(x=x, y=y)
		print("\n%s : %.2f%%" % (model.metrics_names[1],scores[1]*100))


if __name__ == "__main__":
	test = Test()
	dataSet = test.loadData("pima-indians-diabetes.csv")
	test.sequentialModel(dataSet)