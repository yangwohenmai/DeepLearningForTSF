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

# 将list格式数据转换成监督学习数据
def series_to_supervised(data, n_in=1, n_out=1):
	df = DataFrame(data)
	cols = list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
	# 、将位移过的各个列横向拼接到一起
	agg = concat(cols, axis=1)
	print(agg)
	# 删除有null数据的行
	agg.dropna(inplace=True)
	print(agg)
	print(agg.values)
	return agg.values

# 求预测值和真实值之间的均方差
def measure_rmse(actual, predicted):
	return sqrt(mean_squared_error(actual, predicted))

# 训练模型
def model_fit(train, config):
	# 拆分配置信息
	n_input, n_nodes, n_epochs, n_batch = config
	# 将list格式数据转换成监督学习数据
	data = series_to_supervised(train, n_in=n_input)
	train_x, train_y = data[:, :-1], data[:, -1]
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
	# 构造输入数据格式
	x_input = array(history[-n_input:]).reshape(1, n_input)
	# 预测
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
	# step over each time-step in the test set
	for i in range(len(test)):
		# 用训练好的模型开始预测数据，每次传入
		yhat = model_predict(model, history, cfg)
		# store forecast in list of predictions
		predictions.append(yhat)
		# add actual observation to history for the next loop
		history.append(test[i])
	# estimate prediction error
	error = measure_rmse(test, predictions)
	print(test)
	print(predictions)
	print(' > %.3f' % error)
	return error

# repeat evaluation of a config
def repeat_evaluate(data, config, n_test, n_repeats=30):
	# fit and evaluate the model n times
	scores = [walk_forward_validation(data, n_test, config) for _ in range(n_repeats)]
	return scores

# summarize model performance
def summarize_scores(name, scores):
	# print a summary
	scores_m, score_std = mean(scores), std(scores)
	print('%s: %.3f RMSE (+/- %.3f)' % (name, scores_m, score_std))
	# box and whisker plot
	pyplot.boxplot(scores)
	pyplot.show()

series = read_csv('monthly-car-sales.csv', header=0, index_col=0)
data = series.values
# data split
n_test = 12
# 配置信息，[n_input, n_nodes, n_epochs, n_batch]
config = [24, 500, 100, 100]
# grid search
scores = repeat_evaluate(data, config, n_test)
# summarize scores
summarize_scores('mlp', scores)