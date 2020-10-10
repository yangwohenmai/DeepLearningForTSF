# split into standard weeks
from numpy import split
from numpy import array
from pandas import read_csv

# 分割 train/test 集合
def split_dataset(data):
	# 第1行到倒数328行作为训练，第328行到倒数6行作为测试
	train, test = data[1:-328], data[-328:-6]
	# 每7条数据作为一周，分割一次
	train = array(split(train, len(train)/7))
	test = array(split(test, len(test)/7))
	return train, test

# load the new file
dataset = read_csv('household_power_consumption_days.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])
train, test = split_dataset(dataset.values)

# 展示 train/test 数据
print(train.shape)
print(train[0, 0, 0], train[-1, -1, 0])

print(test.shape)
print(test[0, 0, 0], test[-1, -1, 0])