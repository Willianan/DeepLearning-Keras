# -*- coding:utf-8 -*-
"""
@Author：Charles Van
@E-mail:  williananjhon@hotmail.com
@Time：2019/4/9 9:25
@Project:DeepLearning-Keras
@Filename:MLP_regression.py
"""

import numpy as np
from pandas import read_csv
from matplotlib import pyplot as plt
import math
from keras.models import Sequential
from keras.layers import Dense

# 设置参数
seed = 7
batch_size = 2
epochs = 200
fileName = "international-airline-passengers.csv"
footer = 3
look_back = 1


def create_model(dataset):
	dataX, dataY = [], []
	for i in range(len(dataset) - look_back - 1):
		x = dataset[i: i + look_back, 0]
		dataX.append(x)
		y = dataset[i + look_back, 0]
		dataY.append(y)
		print("X: %s,Y: %s" % (x, y))
	return np.array(dataX), np.array(dataY)


# 建立MLP模型
def bulid_model():
	model = Sequential()
	model.add(Dense(units=8, input_dim=look_back, activation='relu'))
	model.add(Dense(units=1))
	model.compile(loss="mean_squared_error", optimizer='adam')
	return model


if __name__ == "__main__":
	# 设定随机种子
	np.random.seed(seed)
	# 导入数据
	data = read_csv(fileName, usecols=[1], engine='python', skipfooter=footer)
	dataSet = data.values.astype('float32')
	train_size = int(len(dataSet) * 0.67)
	validation_size = len(dataSet) - train_size
	train, validation = dataSet[0:train_size, :], dataSet[train_size : len(dataSet), :]
	# 创建dataSet,让数据产生相关性
	x_train, y_train = create_model(train)
	x_validation, y_validation = create_model(validation)

	# 训练模型
	model = bulid_model()
	model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

	# 评估模型
	trainScores = model.evaluate(x_train, y_train, verbose=0)
	print("Train Score: %2.f MSE (%.2f RMSE)" % (trainScores, math.sqrt(trainScores)))
	validationScores = model.evaluate(x_validation, y_validation, verbose=0)
	print("Train Score: %2.f MSE (%.2f RMSE)" % (validationScores, math.sqrt(validationScores)))

	# 图表查看预测趋势
	predict_train = model.predict(x_train)
	predict_validation = model.predict(x_validation)

	# 构建通过训练集进行预测的图表数据
	predict_train_plot = np.empty_like(dataSet)
	predict_train_plot[:, :] = np.nan
	predict_train_plot[look_back:len(predict_train) + look_back, :] = predict_train

	# 构建通过评估数据集进行预测的图表数据
	predict_validation_plot = np.empty_like(dataSet)
	predict_validation_plot[:, :] = np.nan
	predict_validation_plot[len(predict_train) + look_back * 2 + 1:len(dataSet) - 1, :] = predict_validation

	# 图表显示
	plt.plot(dataSet, color='blue')
	plt.plot(predict_train_plot, color='yellow')
	plt.plot(predict_validation_plot, color='red')
	plt.savefig("MLP_regression.png")
	plt.show()
