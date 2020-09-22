from pandas import read_csv
from matplotlib import pyplot
# load all data
data1 = read_csv('datatest.txt', header=0, index_col=1, parse_dates=True, squeeze=True)
data2 = read_csv('datatraining.txt', header=0, index_col=1, parse_dates=True, squeeze=True)
data3 = read_csv('datatest2.txt', header=0, index_col=1, parse_dates=True, squeeze=True)
# determine the number of features
n_features = data1.values.shape[1]
pyplot.figure()
for i in range(1, n_features):
	# specify the subpout
	pyplot.subplot(n_features, 1, i)
	# plot data from each set
	pyplot.plot(data1.index, data1.values[:, i])
	pyplot.plot(data2.index, data2.values[:, i])
	pyplot.plot(data3.index, data3.values[:, i])
	# add a readable name to the plot
	pyplot.title(data1.columns[i], y=0.5, loc='right')
pyplot.show()