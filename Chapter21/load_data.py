# -*- coding:utf-8 -*-
"""
@Author：Charles Van
@E-mail:  williananjhon@hotmail.com
@Time：2019/4/12 9:26
@Project:DeepLearning-Keras
@Filename:load_data.py
"""

from pandas import read_csv
from datetime import datetime
from matplotlib import pyplot as plt

fileName = "pollution_original.csv"


def prase(x):
	return datetime.strptime(x, "%Y %m %d %H")


def load_dataSet():
	dataSet = read_csv(fileName, parse_dates=[['year', 'month', 'day', 'hour']], index_col=0, date_parser=prase)

	# 删除No列
	dataSet.drop('No', axis=1, inplace=True)

	# 设定列名
	dataSet.columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
	dataSet.index.name = 'date'

	# 使用中位数填充缺失值
	dataSet['pollution'].fillna(dataSet['pollution'].mean(), inplace=True)

	return dataSet


if __name__ == "__main__":
	dataSet = load_dataSet()
	print(dataSet.head(5))

	# 查看数据的变化趋势
	groups = [0, 1, 2, 3, 5, 6, 7]
	plt.figure()
	i = 1
	for group in groups:
		plt.subplot(len(groups), 1, i)
		plt.plot(dataSet.values[:, group])
		plt.title(dataSet.columns[group], y=0.5, loc='right')
		i = i + 1
	plt.savefig('polution_original.png')
	plt.show()
