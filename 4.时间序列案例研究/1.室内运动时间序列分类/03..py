# plot series data
from os import listdir
from numpy import array
from numpy import vstack
from numpy.linalg import lstsq
from pandas import read_csv
from matplotlib import pyplot

# return list of traces, and arrays for targets, groups and paths
def load_dataset(prefix=''):
	grps_dir, data_dir = prefix+'groups/', prefix+'dataset/'
	# load mapping files
	targets = read_csv(data_dir + 'MovementAAL_target.csv', header=0)
	groups = read_csv(grps_dir + 'MovementAAL_DatasetGroup.csv', header=0)
	paths = read_csv(grps_dir + 'MovementAAL_Paths.csv', header=0)
	# load traces
	sequences = list()
	target_mapping = None
	for name in listdir(data_dir):
		filename = data_dir + name
		if filename.endswith('_target.csv'):
			continue
		df = read_csv(filename, header=0)
		values = df.values
		sequences.append(values)
	return sequences, targets.values[:,1], groups.values[:,1], paths.values[:,1]

# fit a linear regression function and return the predicted values for the series
def regress(y):
	# define input as the time step
	X = array([i for i in range(len(y))]).reshape(len(y), 1)
	# fit linear regression via least squares
	b = lstsq(X, y)[0][0]
	# predict trend on time step
	yhat = b * X[:,0]
	return yhat

# load dataset
sequences, targets, groups, paths = load_dataset("E:/MyGit/BigDataFile/IndoorMovement/")
# group sequences by paths
paths = [1,2,3,4,5,6]
seq_paths = dict()
for path in paths:
	seq_paths[path] = [sequences[j] for j in range(len(paths)) if paths[j]==path]
# plot one example of a trace for each path
pyplot.figure()
for i in paths:
	pyplot.subplot(len(paths), 1, i)
	# line plot each variable
	for j in [0, 1, 2, 3]:
		pyplot.plot(seq_paths[i][0][:, j], label='Anchor ' + str(j+1))
	pyplot.title('Path ' + str(i), y=0, loc='left')
pyplot.show()
# plot series for a single trace with trend
seq = sequences[0]
variables = [0, 1, 2, 3]
pyplot.figure()
for i in variables:
	pyplot.subplot(len(variables), 1, i+1)
	# plot the series
	pyplot.plot(seq[:,i])
	# plot the trend
	pyplot.plot(regress(seq[:,i]))
pyplot.show()