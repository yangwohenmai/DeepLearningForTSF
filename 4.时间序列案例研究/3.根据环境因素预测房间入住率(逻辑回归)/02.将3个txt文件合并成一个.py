from pandas import read_csv
from pandas import concat
# 加载数据
data1 = read_csv('datatest.txt', header=0, index_col=1, parse_dates=True, squeeze=True)
data2 = read_csv('datatraining.txt', header=0, index_col=1, parse_dates=True, squeeze=True)
data3 = read_csv('datatest2.txt', header=0, index_col=1, parse_dates=True, squeeze=True)
# 合并数据
data = concat([data1, data2, data3])
# 删除列头为no的列数据
data.drop('no', axis=1, inplace=True)
# 保存
data.to_csv('combined.csv')