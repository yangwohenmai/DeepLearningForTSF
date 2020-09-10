# load user movement dataset into memory
from pandas import read_csv
from os import listdir

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
# summarize shape of the loaded data
print(len(sequences), targets.shape, groups.shape, paths.shape)