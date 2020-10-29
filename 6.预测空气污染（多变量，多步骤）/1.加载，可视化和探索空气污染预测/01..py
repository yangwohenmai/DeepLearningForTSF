# load dataset
from numpy import isnan
from numpy import count_nonzero
from pandas import read_csv
# load dataset
dataset = read_csv('E:/MyGit/BigDataFile/dsg-hackathon/TrainingData.csv', header=0)
# summarize
print(dataset.shape)
# trim and transform to floats
values = dataset.values
print(values)
data = values[:, 2:].astype('float32')
print(data)
# summarize amount of missing data
total_missing = count_nonzero(isnan(data))
percent_missing = total_missing / data.size * 100
print('Total Missing: %d/%d (%.1f%%)' % (total_missing, data.size, percent_missing))