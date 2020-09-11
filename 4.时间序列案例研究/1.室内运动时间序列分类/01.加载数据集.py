from pandas import read_csv
from os import listdir

# return list of traces, and arrays for targets, groups and paths
def load_dataset(prefix=''):
	# 拼接文件路径所在路径
	grps_dir, data_dir = prefix+'groups/', prefix+'dataset/'
	# 根据路径加载文件
	targets = read_csv(data_dir + 'MovementAAL_target.csv', header=0)
	groups = read_csv(grps_dir + 'MovementAAL_DatasetGroup.csv', header=0)
	paths = read_csv(grps_dir + 'MovementAAL_Paths.csv', header=0)

	# 加载文件夹下的所有特定文件
	sequences = list()
	for name in listdir(data_dir):
		filename = data_dir + name
		# 只加载以_target.csv结尾的文件
		if filename.endswith('_target.csv'):
			continue
		df = read_csv(filename, header=0)
		values = df.values
		sequences.append(values)
	return sequences, targets.values[:,1], groups.values[:,1], paths.values[:,1]

# 加载数据
sequences, targets, groups, paths = load_dataset("E:/MyGit/BigDataFile/IndoorMovement/")
# 打印加载文件的shape
print(len(sequences), targets.shape, groups.shape, paths.shape)