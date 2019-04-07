# -*- coding:utf-8 -*-
"""
@Author：Charles Van
@E-mail:  williananjhon@hotmail.com
@Time：2019/4/1 21:37
@Project:DeepLearning-Keras
@Filename:Bank_Marketing.py
"""

import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# 导入数据并分类转化为数字
dataSet = pd.read_csv("bank.csv", delimiter=';')
dataSet['job'] = dataSet['job'].replace(to_replace=['admin.', 'unknown', 'unemployed', 'management',
                                                    'housemaid', 'entrepreneur', 'student', 'blue-collar',
                                                    'self-employed', 'retired', 'technician', 'services'],
                                        value=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
dataSet['marital'] = dataSet['marital'].replace(to_replace=['married', 'single', 'divorced'], value=[0, 1, 2])
dataSet['education'] = dataSet['education'].replace(to_replace=['unknown', 'secondary', 'primary', 'tertiary'],
                                                    value=[0, 2, 1, 3])
dataSet['default'] = dataSet['default'].replace(to_replace=['no', 'yes'], value=[0, 1])
dataSet['housing'] = dataSet['housing'].replace(to_replace=['no', 'yes'], value=[0, 1])
dataSet['loan'] = dataSet['loan'].replace(to_replace=['no', 'yes'], value=[0, 1])
dataSet['contact'] = dataSet['contact'].replace(to_replace=['cellular', 'unknown', 'telephone'], value=[0, 1, 2])
dataSet['poutcome'] = dataSet['poutcome'].replace(to_replace=['unknown', 'other', 'success', 'failure'],
                                                  value=[0, 1, 2, 3])
dataSet['month'] = dataSet['month'].replace(to_replace=['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                                                        'jul', 'aug', 'sep', 'oct', 'nov', 'dec'],
                                            value=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
dataSet['y'] = dataSet['y'].replace(to_replace=['no', 'yes'], value=[0, 1])

# 分离输入和输出
array = dataSet.values
x = array[:, 0:16]
Y = array[:, 16]

# 设置随机种子
seed = 7
np.random.seed(seed)


# 构建模型函数
def create_model(units_list=[16], optimizer='adam', init='normal'):
	# 构建模型
	model = Sequential()
	# 构建第一个隐藏层和输入层
	model.add(Dense(units=units_list[0], activation='relu', input_dim=16, kernel_initializer=init))
	# 构建更多隐藏层
	for units in units_list[1:]:
		model.add(Dense(units=units, activation='relu', kernel_initializer=init))
	model.add(Dense(units=1, activation='sigmoid', kernel_initializer=init))
	# 编译模型
	model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
	return model


model = KerasClassifier(build_fn=create_model, epochs=200, batch_size=5, verbose=0)
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(model, x, Y, cv=kfold)
print('Accuracy: %.2f%% (%.2f)' % (results.mean() * 100, results.std()))

new_x = StandardScaler().fit_transform(x)
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(model, new_x, Y, cv=kfold)
print('Accuracy: %.2f%% (%.2f)' % (results.mean() * 100, results.std()))

# 调参选择最优模型
param_grid = {}
param_grid['units_list'] = [[16], [30], [16, 8], [30, 8]]
# 调参
grid = GridSearchCV(estimator=model, param_grid=param_grid)
results = grid.fit(new_x, Y)

# 输出结果
print('Best: %f using %s' % (results.best_score_, results.best_params_))
means = results.cv_results_['mean_test_score']
stds = results.cv_results_['std_test_score']
params = results.cv_results_['params']

for mean, std, param in zip(means, stds, params):
	print('%f (%f) with: %r' % (mean, std, param))