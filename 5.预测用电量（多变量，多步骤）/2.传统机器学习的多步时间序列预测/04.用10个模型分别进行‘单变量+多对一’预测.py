# recursive multi-step forecast with linear algorithms
from math import sqrt
from numpy import split
from numpy import array
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import Lars
from sklearn.linear_model import LassoLars
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import SGDRegressor

# 将数据按week分割成训练集和测试集
def split_dataset(data):
	# 分割成训练集和测试集
	train, test = data[1:-328], data[-328:-6]
	# 分割成以周为单位
	train = array(split(train, len(train)/7))
	test = array(split(test, len(test)/7))
	return train, test

# 评估预测结果
def evaluate_forecasts(actual, predicted):
	scores = list()
	# 评估预测出来7列数据，每一列数据的均方差
	for i in range(actual.shape[1]):
		# calculate mse
		mse = mean_squared_error(actual[:, i], predicted[:, i])
		# calculate rmse
		rmse = sqrt(mse)
		# store
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

# 创建模型列表，后续循环调用模型
def get_models(models=dict()):
	models['lr'] = LinearRegression()
	models['lasso'] = Lasso()
	models['ridge'] = Ridge()
	models['en'] = ElasticNet()
	models['huber'] = HuberRegressor()
	models['lars'] = Lars()
	models['llars'] = LassoLars()
	models['pa'] = PassiveAggressiveRegressor(max_iter=1000, tol=1e-3)
	models['ranscac'] = RANSACRegressor()
	models['sgd'] = SGDRegressor(max_iter=1000, tol=1e-3)
	print('Defined %d models' % len(models))
	return models

# create a feature preparation pipeline for a model
def make_pipeline(model):
	steps = list()
	# 数据标准化
	steps.append(('standardize', StandardScaler()))
	# 数据归一化
	steps.append(('normalize', MinMaxScaler()))
	# the model
	steps.append(('model', model))
	# create pipeline
	pipeline = Pipeline(steps=steps)
	return pipeline

# 进行递归多步预测,所谓递归预测即用新预测出的值作为输入，去预测后续值
def forecast(model, input_x, n_input):
    yhat_sequence = list()
    # 传入训练集最后一条数据，作为测试集的第一条输入
    input_data = [x for x in input_x]
    for j in range(7):
        # 取input_data列表的最后7条数据，X -> reshape(1,7)
        X = array(input_data[-n_input:]).reshape(1, n_input)
        # 预测并保存结果
        yhat = model.predict(X)[0]
        yhat_sequence.append(yhat)
        # 预测结果加入列表，用于下递归次预
        input_data.append(yhat)
    return yhat_sequence

# 将多维week数据取出第一列，并转成一维序列
def to_series(data):
	# data数据集中一共有159周数据，每周7天，每天8个特征值：(159,7,8)
	# 当前函数提取data中每一周的每一天的第一个特征值,得到：(159,7)
	# data = [array([[]])]->(159,7,8)
	# series = [array([])]->(159,7)
	series = [week[:, 0] for week in data]
	# 将series展平成一维(159*7,) 即：series = array([])->(1113,)
	series = array(series).flatten()
	return series

# 构造“多对一(7->1)”的监督学习型数据的 输入 输出
def to_supervised(history, n_step):
	# 获取history集合中的一列的集合
	data = to_series(history)
	X, y = list(), list()
	ix_start = 0
	# 遍历数据，构建(7->1)”的监督学习型数据
	for i in range(len(data)):
		#定义每次截取数据的起始和结尾位置
		ix_end = ix_start + n_step
		# 顺序截取7条数据作为输入，1条数据做输出
		if ix_end < len(data):
			X.append(data[ix_start:ix_end])
			y.append(data[ix_end])
		# 移动到下一个时间步起始位置
		ix_start += 1
	return array(X), array(y)

# 对模型先拟合，再预测
def sklearn_predict(model, history, n_step):
    # 将数据转换成7->1的监督数据，train_x(1106,7),train_y(1106)
    train_x, train_y = to_supervised(history, n_step)
    # 创建一个pipeline模型列表
    pipeline = make_pipeline(model)
    # 拟合
    pipeline.fit(train_x, train_y)
    # 预测,将训练集最后一条数据作为测试集的第一条输入
    yhat_sequence = forecast(pipeline, train_x[-1, :], n_step)
    return yhat_sequence

# 拟合、预测、评估模型
def evaluate_model(model, train, test, n_input):
    # 获取array类型的训练集数据history = [array([[]])]->(159,7,8)
    history = [x for x in train]
    predictions = list()
    # 预测测试集中对应的每条数据
    for i in range(len(test)):
        # 对模型先拟合，再预测
        yhat_sequence = sklearn_predict(model, history, n_input)
        # 保存预测结果
        predictions.append(yhat_sequence)
        # get real observation and add to history for predicting the next week
        # history.append(test[i, :])
    predictions = array(predictions)
    # 评估预测结果
    score, scores = evaluate_forecasts(test[:, :, 0], predictions)
    return score, scores

# load the new file
dataset = read_csv('household_power_consumption_days.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])
# 将数据按week分割成训练集和测试集
train, test = split_dataset(dataset.values)
# 创建模型列表
models = get_models()
n_input = 7
days = ['sun', 'mon', 'tue', 'wed', 'thr', 'fri', 'sat']
# 对每个模型进行评估
for name, model in models.items():
	# 拟合、预测、评估模型
	score, scores = evaluate_model(model, train, test, n_input)
	# 统计得分
	summarize_scores(name, score, scores)
	# 画出得分图
	pyplot.plot(days, scores, marker='o', label=name)
# show plot
pyplot.legend()
#pyplot.show()