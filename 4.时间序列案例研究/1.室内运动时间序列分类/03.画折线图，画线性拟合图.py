from os import listdir
from numpy import array
from numpy import vstack
from numpy.linalg import lstsq
from pandas import read_csv
from matplotlib import pyplot

# 加载traces, arrays for targets, groups, paths文件的数据
def load_dataset(prefix=''):
	grps_dir, data_dir = prefix+'groups/', prefix+'dataset/'
	# load mapping files
	targets = read_csv(data_dir + 'MovementAAL_target.csv', header=0)
	groups = read_csv(grps_dir + 'MovementAAL_DatasetGroup.csv', header=0)
	paths = read_csv(grps_dir + 'MovementAAL_Paths.csv', header=0)
	# load traces
	sequences = list()
	for name in listdir(data_dir):
		filename = data_dir + name
		if filename.endswith('_target.csv'):
			continue
		df = read_csv(filename, header=0)
		values = df.values
		sequences.append(values)
	return sequences, targets.values[:,1], groups.values[:,1], paths.values[:,1]

# 最小二乘法拟合线性回归模型，并预测每个时间步的输出，并返回捕获数据趋势的序列
def regress(y):
	# define input as the time step
	X = array([i for i in range(len(y))]).reshape(len(y), 1)
	# 用最小二乘解返回线性矩阵方程
	b = lstsq(X, y)[0][0]
	# predict trend on time step
	yhat = b * X[:,0]
	return yhat

# 加载数据
sequences, targets, groups, paths = load_dataset("E:/MyGit/BigDataFile/IndoorMovement/")
# group sequences by paths
paths = [1,2,3,4,5,6]
seq_paths = dict()
for path in paths:
	seq_paths[path] = [sequences[j] for j in range(len(paths)) if paths[j]==path]
	
# 绘制数据的折线图
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


# 绘制数据的线性拟合图
variables = [0, 1, 2, 3]
pyplot.figure()
for i in variables:
	pyplot.subplot(len(variables), 1, i+1)
	# plot the series
	pyplot.plot(seq[:,i])
	# plot the trend
	pyplot.plot(regress(seq[:,i]))
pyplot.show()

import numpy as np
from matplotlib import pyplot
x=np.array([0,1,2,3])
y=np.array([3,2,1,0])
A=np.vstack([x,np.ones(len(x))]).T
k,c=np.linalg.lstsq(A,y,rcond=None)[0]
pyplot.plot(x,y,'o',label='原始data',markersize=10)
pyplot.plot(x,k*x+c,'r',label='拟合曲线line')
pyplot.legend()

pyplot.show()