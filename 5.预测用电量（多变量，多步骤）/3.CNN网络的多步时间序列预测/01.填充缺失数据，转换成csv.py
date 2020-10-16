# load and clean-up data
from numpy import nan
from numpy import isnan
from pandas import read_csv
from pandas import to_numeric

# 遍历 用前一天数据填充缺失数据
def fill_missing(values):
  # 数据是分钟级别，60*24是前一天同一时间点的数据
	one_day = 60 * 24
	# 遍历每一个元素
	for row in range(values.shape[0]):
		for col in range(values.shape[1]):
		  # 如果数据为空，用前一天同一时间点的数据经停填充
			if isnan(values[row, col]):
				values[row, col] = values[row - one_day, col]

# load all data
dataset = read_csv('E:\MyGit\BigDataFile\household_power_consumption\household_power_consumption.txt', sep=';', header=0, low_memory=False, infer_datetime_format=True, parse_dates={'datetime':[0,1]}, index_col=['datetime'])
# 将缺失数据标记位NaN
dataset.replace('?', nan, inplace=True)
dataset = dataset.astype('float32')

# 遍历填充缺失数据
fill_missing(dataset.values)
# 所有行：第一列 减去5,6,7列
values = dataset.values
dataset['sub_metering_4'] = (values[:,0] * 1000 / 60) - (values[:,4] + values[:,5] + values[:,6])
# 保存
dataset.to_csv('household_power_consumption.csv')