# -*- coding:utf-8 -*-
"""
@Author：Charles Van
@E-mail:  williananjhon@hotmail.com
@Time：2019/4/1 10:28
@Project:DeepLearning-Keras
@Filename:cross_validation.py
"""

from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import StratifiedKFold


class CrossValidation:
	def loadDataSet(self, filename):
		return np.loadtxt(filename, delimiter=',')

	def CrossValidationModel(self, dataSet):
		np.random.seed(7)
		x = dataSet[:, 0:8]
		Y = dataSet[:, 8]
		Kfold = StratifiedKFold(n_splits=10, random_state=7, shuffle=True)
		cvscores = []
		for train, validation in Kfold.split(x, Y):
			model = Sequential()
			model.add(Dense(12, input_dim=8, activation='relu'))
			model.add(Dense(8, activation='relu'))
			model.add(Dense(1, activation='sigmoid'))
			model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
			model.fit(x[train], Y[train], epochs=150, batch_size=10, verbose=0)
			scores = model.evaluate(x[validation], Y[validation], verbose=0)
			print("%s : %.2f%%" % (model.metrics_names[1], scores[1] * 100))
			cvscores.append(scores[1] * 100)
		print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores),np.std(cvscores)))


if __name__ == "__main__":
	test = CrossValidation()
	dataSet = test.loadDataSet("pima-indians-diabetes.csv")
	test.CrossValidationModel(dataSet)