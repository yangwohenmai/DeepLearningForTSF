# load dataset
from numpy import isnan
from numpy import count_nonzero
from pandas import read_csv
import numpy

#dataset = read_csv('E:/MyGit/BigDataFile/dsg-hackathon/TrainingData.csv', header=0)
dataset = read_csv(r'D:\咗MyGit\BigDataFile\dsg-hackathon\TrainingData.csv', header=0)
# summarize
print(dataset.shape)
# trim and transform to floats
values = dataset.values
data = values[:, 6:].astype('float32')
# summarize amount of missing data
total_missing = count_nonzero(isnan(data))
percent_missing = total_missing / data.size * 100
print('Total Missing: %d/%d (%.1f%%)' % (total_missing, data.size, percent_missing))