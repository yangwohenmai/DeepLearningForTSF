# evaluate mlp
from math import sqrt
from numpy import array
from numpy import mean
from numpy import std
from pandas import DataFrame
from pandas import concat
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot

# 把单列数据按给定数n_test，拆分成测试集和训练集
def train_test_split(data, n_test):
	return data[:-n_test], data[-n_test:]

# 将单列list格式数据转换成（输入/输出）监督学习数据
# data：单列数据
# n_in：用于构造输入序列(t-n, ... t-1)，n_in表示每行监督学习数据的长度，如n_in=9，可构造8->1。n_in=0表示停用
# n_out：用于构造输出序列(t, t+1, ... t+n)，n_out表示每行监督学习数据的长度，如n_in=9，可构造8->1。n_out=0表示停用
def series_to_supervised(data, n_in, n_out=0):
	df = DataFrame(data)
	cols = list()
	# 得到(t-n, ... t-1, t)：从n_in到-1循环，步长为-1。每次将data向下移动i行，将移动过的数据添加到cols数组中
	for i in range(n_in, -1, -1):
		cols.append(df.shift(i))
	# 得到(t, t+1, ... t+n)：默认n_out=0，
	for i in range(0, n_out):
		cols.append(df.shift(-i))
	# 将位移过的各个列，横向拼接到一起
	agg = concat(cols, axis=1)
	# 删除带有null数据的行
	agg.dropna(inplace=True)
	# 每一行的前n-1列作为输入值，最后一列作为输出值
	return agg.values[:, :-1], agg.values[:, -1]

# 求预测值和真实值之间的均方差
def measure_rmse(actual, predicted):
	return sqrt(mean_squared_error(actual, predicted))

# 训练模型
def model_fit(train, config):
	# 拆分配置信息
	n_input, n_nodes, n_epochs, n_batch = config
	# 将list格式数据转换成监督学习数据,并得到输入集和输出集
	train_x, train_y = series_to_supervised(train, n_input)
	# define model
	model = Sequential()
	model.add(Dense(n_nodes, activation='relu', input_dim=n_input))
	model.add(Dense(1))
	model.compile(loss='mse', optimizer='adam')
	# fit
	model.fit(train_x, train_y, epochs=n_epochs, batch_size=n_batch, verbose=0)
	return model

# 用训练好的模型model开始预测
def model_predict(model, history, config):
	# 拆分配置信息
	n_input, _, _, _ = config
	# 取训练数据的最后24条，作为预测数据的初始数据
	x_input = array(history[-n_input:]).reshape(1, n_input)
	yhat = model.predict(x_input, verbose=0)
	return yhat[0]

# walk-forward validation for univariate data
def walk_forward_validation(data, n_test, cfg):
	predictions = list()
	# 将数据拆分成测试集和训练集
	train, test = data[:-n_test], data[-n_test:]
	# 训练模型
	model = model_fit(train, cfg)
	# seed history with training dataset
	history = [x for x in train]
	# 开始单步循环预测数据
	for i in range(len(test)):
		# 用训练好的模型开始预测数据，每次传入一个新的history
		yhat = model_predict(model, history, cfg)
		# 将预测出的值存入预测队列
		predictions.append(yhat)
		# 将一个新的观测值插入输入数据，进行下一轮预测
		history.append(test[i])
	# 评估预测结果的误差
	error = measure_rmse(test, predictions)
	print(' > %.3f' % error)
	return error

# 对模型重复运行30次，记录每次预测结果的均方差
def repeat_evaluate(data, config, n_test, n_repeats=30):
	# fit and evaluate the model n times
	scores = [walk_forward_validation(data, n_test, config) for _ in range(n_repeats)]
	return scores

# 汇总模型均方差
def summarize_scores(name, scores):
	# 打印出均值，标准差
	scores_m, score_std = mean(scores), std(scores)
	print('%s: %.3f RMSE (+/- %.3f)' % (name, scores_m, score_std))
	# 画出箱须图
	pyplot.boxplot(scores)
	pyplot.show()

series = read_csv('monthly-car-sales.csv', header=0, index_col=0)
data = series.values
# data split
n_test = 12
# 配置信息，[输入长度n_input, 节点数n_nodes, 周期n_epochs, 批次n_batch]
config = [5, 500, 100, 100]
# grid search
scores = repeat_evaluate(data, config, n_test)
# 汇总模型均方差
summarize_scores('mlp', scores)