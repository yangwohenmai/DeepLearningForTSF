from math import sqrt
from numpy import split
from numpy import array
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

# 将数据按week分割成训练集和测试集
def split_dataset(data):
	# 分割成训练集和测试集
	train, test = data[1:-328], data[-328:-6]
	# 分割成以周为单位
	train = array(split(train, len(train)/7))
	test = array(split(test, len(test)/7))
	return train, test

# 评估真实值和预测值的RMSE
def evaluate_forecasts(actual, predicted):
	scores = list()
	# 评估预测出来7列数据，每一列数据的均方差
	for i in range(actual.shape[1]):
		mse = mean_squared_error(actual[:, i], predicted[:, i])
		rmse = sqrt(mse)
		scores.append(rmse)
	# calculate overall RMSE
	s = 0
	for row in range(actual.shape[0]):
		for col in range(actual.shape[1]):
			s += (actual[row, col] - predicted[row, col])**2
	score = sqrt(s / (actual.shape[0] * actual.shape[1]))
	return score, scores

# 统计得分
def summarize_scores(name, score, scores):
	s_scores = ', '.join(['%.1f' % s for s in scores])
	print('%s: [%.3f] %s' % (name, score, s_scores))

# 构造“多对多(7->7)”的监督学习型数据的 输入 输出
def to_supervised(train, n_input, n_out=7):
	# 将3维数据(159,7,8)展平成2维(1113,8)
	data = train.reshape((train.shape[0]*train.shape[1], train.shape[2]))
	X, y = list(), list()
	in_start = 0
	# 遍历数据，构建“多对多(7->7)”的监督学习型数据
	for _ in range(len(data)):
		#定义每次 输入 截取数据的起始和结尾位置
		in_end = in_start + n_input
		#定义每次 输出 截取数据的起始和结尾位置
		out_end = in_end + n_out
		# 遍历数据集,构建(7->7)的监督学习数据
		if out_end <= len(data):
			# 输入X：eg:(0~7)、(1~8)、(2~9)、(3~10)
			x_input = data[in_start:in_end, 0]
			x_input = x_input.reshape((len(x_input), 1))
			X.append(x_input)
			# 输出Y：eg:(7~14)、(8~15)、(9~16)、(10~17)
			y.append(data[in_end:out_end, 0])
		# 移动到下一个时间步起始位置
		in_start += 1
	return array(X), array(y)

# 构建网络模型
def build_model(train, n_input):
	# 构造“多对多(7->7)”的监督学习型数据
	train_x, train_y = to_supervised(train, n_input)
	# 日志输出0，训练次数20，批次大小4，时间步为7，特征值为1，每次预测7条数据
	verbose, epochs, batch_size = 0, 20, 4
	n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
	# define model
	model = Sequential()
	# 时间步为7，特征值为1，每次预测7条数据
	model.add(Conv1D(filters=16, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features)))
	model.add(MaxPooling1D(pool_size=2))
	model.add(Flatten())
	model.add(Dense(10, activation='relu'))
	model.add(Dense(n_outputs))
	model.compile(loss='mse', optimizer='adam')
	# fit network
	model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
	return model

# make a forecast
def forecast(model, history, n_input):
	data = array(history)
	# 将3维数据(159,7,8)展平成2维(1113,8)
	data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
	# 取训练集最后7条数据，作为测试集的第首次输入
	input_x = data[-n_input:, 0]
	# 重构输入形状(7,)->(1,7,1)
	input_x = input_x.reshape((1, len(input_x), 1))
	# 预测未来一周值(1,7)，并返回需要的格式(7,)
	yhat = model.predict(input_x, verbose=0)
	yhat = yhat[0]
	return yhat

# 拟合、预测、评估模型
def evaluate_model(train, test, n_input):
	# fit model
	model = build_model(train, n_input)
	# 获取array类型的训练集数据history = [array([[]])]->(159,7,8)
	history = [x for x in train]
	predictions = list()
	# 预测测试集中对应的每条数据
	for i in range(len(test)):
		# 按周(每次预测7条)进行预测，并保存预测结果，用于后续递归预测
		yhat_sequence = forecast(model, history, n_input)
		predictions.append(yhat_sequence)
	# 评估预测结果
	predictions = array(predictions)
	score, scores = evaluate_forecasts(test[:, :, 0], predictions)
	return score, scores


dataset = read_csv('household_power_consumption_days.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])
# # 将数据按week分割成训练集[159,7,8]和测试集[46,7,8]
train, test = split_dataset(dataset.values)
# evaluate model and get scores
n_input = 7
score, scores = evaluate_model(train, test, n_input)
# 统计得分
summarize_scores('cnn', score, scores)
# 画出得分图
days = ['sun', 'mon', 'tue', 'wed', 'thr', 'fri', 'sat']
pyplot.plot(days, scores, marker='o', label='cnn')
pyplot.show()