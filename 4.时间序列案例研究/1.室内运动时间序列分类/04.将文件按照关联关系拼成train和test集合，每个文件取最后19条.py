# prepare fixed length vector dataset
from os import listdir
from numpy import array
from numpy import savetxt
from pandas import read_csv

# 加载IndoorMovement/dataset和IndoorMovement/groups下的所有文件
def load_dataset(prefix=''):
	grps_dir, data_dir = prefix+'groups/', prefix+'dataset/'
	# 读取单个文件，MovementAAL_target作数输出数据
	targets = read_csv(data_dir + 'MovementAAL_target.csv', header=0)
	groups = read_csv(grps_dir + 'MovementAAL_DatasetGroup.csv', header=0)
	paths = read_csv(grps_dir + 'MovementAAL_Paths.csv', header=0)
	# 加载IndoorMovement/dataset文件夹下除了以_target.csv结尾以外的文件，作为输入数据
	sequences = list()
	for name in listdir(data_dir):
		filename = data_dir + name
		if filename.endswith('_target.csv'):
			continue
		df = read_csv(filename, header=0)
		values = df.values
		# 将每个文件里的数据加载到sequences中
		sequences.append(values)
	return sequences, targets.values[:,1], groups.values[:,1], paths.values[:,1]

# 将加载的输入文件RSS_1和输出文件_target构建成一维的输入输出向量
def create_dataset(sequences, targets):
	# 用于存储转换后的数据
	transformed = list()
	# 每行数据有4列
	n_vars = 4
	# 所有文件中最少的一个文件只有19行数据，所以每个文件只取最后19行数据，统一“输入数据”维度
	n_steps = 19
	# process each trace in turn
	for i in range(len(sequences)):
		seq = sequences[i]
		vector = list()
		# 从后向前取n_steps=19行数据，作为输入
		for row in range(1, n_steps+1):
			# 每行有n_vars=4列数据
			for col in range(n_vars):
				vector.append(seq[-row, col])
		# 从targets文件中添加输出
		vector.append(targets[i])
		# 将构建好的“输入->输出”对，加载到transformed中
		transformed.append(vector)
	# 将存储数组的列表 转换成 存储数组的数组
	transformed = array(transformed)
	transformed = transformed.astype('float32')
	return transformed

# 加载数据
sequences, targets, groups, paths = load_dataset("IndoorMovement/")
# 根据MovementAAL_DatasetGroup.csv文件中的序号，将sequences数据分成seq1、seq2、seq3三组
seq1 = [sequences[i] for i in range(len(groups)) if groups[i]==1]
seq2 = [sequences[i] for i in range(len(groups)) if groups[i]==2]
seq3 = [sequences[i] for i in range(len(groups)) if groups[i]==3]
# 根据MovementAAL_DatasetGroup.csv文件中的序号，将targets数据分成targets1、targets2、targets3三组
targets1 = [targets[i] for i in range(len(groups)) if groups[i]==1]
targets2 = [targets[i] for i in range(len(groups)) if groups[i]==2]
targets3 = [targets[i] for i in range(len(groups)) if groups[i]==3]
# 将seq1+seq2作为输入数据，targets1+targets2作为输出数据，构建一维训练数据
es2_train = create_dataset(seq1+seq2, targets1+targets2)
# 将seq3作为输入数据，targets3作为输出数据，构建一维测试数据
es2_test = create_dataset(seq3, targets3)
print('ES2 Train: %s' % str(es2_train.shape))
print('ES2 Test: %s' % str(es2_test.shape))
savetxt('es2_train.csv', es2_train, delimiter=',')
savetxt('es2_test.csv', es2_test, delimiter=',')