# summarize simple information about user movement data
from os import listdir
from numpy import array
from numpy import vstack
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

# load dataset
sequences, targets, groups, paths = load_dataset("E:/MyGit/BigDataFile/IndoorMovement/")
# summarize class breakdown
class1,class2 = len(targets[targets==-1]), len(targets[targets==1])
print('Class=-1: %d %.3f%%' % (class1, class1/len(targets)*100))
print('Class=+1: %d %.3f%%' % (class2, class2/len(targets)*100))
# histogram for each anchor point
all_rows = vstack(sequences)
pyplot.figure()
variables = [0, 1, 2, 3]
for v in variables:
	pyplot.subplot(len(variables), 1, v+1)
	pyplot.hist(all_rows[:, v], bins=20)
pyplot.show()
# histogram for trace lengths
trace_lengths = [len(x) for x in sequences]
pyplot.hist(trace_lengths, bins=50)
pyplot.show()