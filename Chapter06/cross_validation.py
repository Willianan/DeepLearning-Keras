# -*- coding:utf-8 -*-
"""
@Author：Charles Van
@E-mail:  williananjhon@hotmail.com
@Time：2019/4/1 11:09
@Project:DeepLearning-Keras
@Filename:cross_validation.py
"""

from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from keras.wrappers.scikit_learn import KerasClassifier

class CrossValidation:
	def loadDataSet(self,filename):
		return np.loadtxt(filename,delimiter=',')

	def create_model(self):
		model = Sequential()
		model.add(Dense(units=12,input_dim=8,activation='relu'))
		model.add(Dense(units=8,activation='relu'))
		model.add(Dense(units=1,activation='sigmoid'))
		model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
		return model

	def KerasClass(self,dataSet):
		seed = 7
		np.random.seed(seed)
		x = dataSet[:,0:8]
		Y = dataSet[:,8]
		model = KerasClassifier(build_fn=self.create_model,epochs=150,batch_size=10,verbose = 0)
		kfold = StratifiedKFold(n_splits=10,shuffle=True,random_state=seed)
		result = cross_val_score(model,x,Y,cv=kfold)
		print(result.mean())


if __name__ == "__main__":
	test = CrossValidation()
	dataSet = test.loadDataSet("pima-indians-diabetes.csv")
	test.KerasClass(dataSet)