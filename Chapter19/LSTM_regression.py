# -*- coding:utf-8 -*-
"""
@Author：Charles Van
@E-mail:  williananjhon@hotmail.com
@Time：2019/4/9 10:22
@Project:DeepLearning-Keras
@Filename:LSTM_regression.py
"""


import numpy as np
from matplotlib import pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense,LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

seed = 7
batch_size = 1
epochs = 100
fileName = 'international-airline-passengers.csv'
footer = 3
look_back = 1

def create_dataSet(dataSet):
	dataX,dataY = [],[]
	for i in range(len(dataSet) - look_back - 1):
		x = dataSet[i:i+look_back,0]
		dataX.append(x)
		y = dataSet[i + look_back,0]
		dataY.append(y)
		print("X: %s,Y: %s" % (x,y))
	return np.array(dataX),np.array(dataY)

def create_model():
	model = Sequential()
	model.add(LSTM(units=4,input_shape=(1,look_back)))
	model.add(Dense(units=1))
	model.compile(loss='mean_squared_error',optimizer='adam')
	return model

if __name__ == "__main__":
	# 设置随机数种子
	np.random.seed(seed)
	# 导入数据
	data = read_csv(fileName,usecols=[1],engine='python',skipfooter=footer)
	dataSet = data.values.astype('float32')
	# 标准化数据
	scaler = MinMaxScaler()
	dataSet = scaler.fit_transform(dataSet)
	train_size = int(len(dataSet) * 0.67)
	validation_size = len(dataSet) - train_size
	train,validation = dataSet[0:train_size,:],dataSet[train_size:len(dataSet),:]

	#创建dataSet，让数据产生相关性
	x_train,y_train = create_dataSet(train)
	x_validation,y_validation = create_dataSet(validation)

	# 将输入转化成[样本，时间步长，特征]
	x_train = np.reshape(x_train,(x_train.shape[0],1,x_train.shape[1]))
	x_validation = np.reshape(x_validation,(x_validation.shape[0],1,x_validation.shape[1]))

	# 训练模型
	model = create_model()
	model.fit(x_train,y_train,epochs=epochs,batch_size=batch_size,verbose=2)

	# 模型预测数据
	predict_train = model.predict(x_train)
	predict_validation = model.predict(x_validation)

	# 反标准化数据，目的是保证MSE的准确性
	predict_train = scaler.inverse_transform(predict_train)
	y_train = scaler.inverse_transform([y_train])
	predict_validation = scaler.inverse_transform(predict_validation)
	y_validation = scaler.inverse_transform([y_validation])

	# 评估模型
	train_score = math.sqrt(mean_squared_error(y_train[0],predict_train[:,0]))
	print("Train Score : %.2f RMSE" % train_score)
	validation_score = math.sqrt(mean_squared_error(y_validation[0],predict_validation[:,0]))
	print("Validation Score : %.2f RMSE" % validation_score)

	# 构建通过训练数据集进行预测的图表数据
	predict_train_plot = np.empty_like(dataSet)
	predict_train_plot[:,:] = np.nan
	predict_train_plot[look_back:len(predict_train) + look_back,:] = predict_train

	# 构建通过评估数据集进行预测的图表数据
	predict_validation_plot = np.empty_like(dataSet)
	predict_validation_plot[:,:] = np.nan
	predict_validation_plot[len(predict_train) + look_back * 2 + 1:len(dataSet) - 1,:] = predict_validation

	# 图表显示
	dataSet = scaler.inverse_transform(dataSet)
	plt.plot(dataSet,color = 'blue')
	plt.plot(predict_train_plot,color = 'green')
	plt.plot(predict_validation_plot,color='red')
	plt.savefig("LSTM_regression.png")
	plt.show()