# 将分钟级别的数据合并成不同级别，确实的日期中间补零
from pandas import read_csv
# 加载数据
dataset = read_csv('household_power_consumption2.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])

# 根据年级别合并数据
#daily_groups = dataset.resample('Y')
# 根据月级别合并数据
#daily_groups = dataset.resample('M')
# 根据日级别合并数据
daily_groups = dataset.resample('D')
daily_data = daily_groups.sum()

# 展示
print(daily_data.shape)
print(daily_data.head())

# 保存
daily_data.to_csv('household_power_consumption_days2.csv')