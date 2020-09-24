from pandas import read_csv
from numpy import mean
from numpy import std
from numpy import delete
from numpy import savetxt
# load the dataset.
data = read_csv('EEG_Eye_State.csv', header=None)
values = data.values
# 每次处理一列数据
for i in range(values.shape[1] - 1):
	# 计算每列数据的均值和标准差
	data_mean, data_std = mean(values[:,i]), std(values[:,i])
	# 定义异常值的上界和下界为4倍标准差
	cut_off = data_std * 4
	lower, upper = data_mean - cut_off, data_mean + cut_off
	# 剔除过小的极值
	too_small = [j for j in range(values.shape[0]) if values[j,i] < lower]
	values = delete(values, too_small, 0)
	print('>deleted %d rows' % len(too_small))
	# 剔除过大的极值
	too_large = [j for j in range(values.shape[0]) if values[j,i] > upper]
	values = delete(values, too_large, 0)
	print('>deleted %d rows' % len(too_large))
# 保存处理后的文件
savetxt('EEG_Eye_State_no_outliers.csv', values, delimiter=',')