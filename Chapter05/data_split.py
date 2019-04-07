# -*- coding:utf-8 -*-
"""
@Author：Charles Van
@E-mail:  williananjhon@hotmail.com
@Time：2019/4/1 10:45
@Project:DeepLearning-Keras
@Filename:data_split.py
"""

from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np

class DataSpiltModel:
	def loadDataSet(self,filename):
		return np.loadtxt(filename,delimiter=',')

	def DataSplitModel(self,dataSet):
		seed = 7
		np.random.seed(seed)
		x = dataSet[:,0:8]
		Y = dataSet[:,8]
		x_train,x_validation,Y_train,Y_validation = train_test_split(x,Y,test_size=0.2,random_state=seed)
		model = Sequential()
		model.add(Dense(12,input_dim=8,activation='relu'))
		model.add(Dense(8,activation='relu'))
		model.add(Dense(1,activation='sigmoid'))
		model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
		model.fit(x_train,Y_train,validation_data=(x_validation,Y_validation),epochs=150,batch_size=10)


if __name__ == "__main__":
	test = DataSpiltModel()
	dataSet = test.loadDataSet("pima-indians-diabetes.csv")
	test.DataSplitModel(dataSet)