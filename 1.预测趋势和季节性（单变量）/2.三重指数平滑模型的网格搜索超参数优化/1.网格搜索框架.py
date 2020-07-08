"""
一个简单的网格搜索框架
网格搜索就是穷举法，对所有可能的参数组合都带入程序，进行尝试。
模型参数对应：SARIMA(p,d,q)(P,D,Q)m，对于模型来说并不是所有输入参数都是有效的，
如季节周期参数m不能为0，当m=0时，会导致SARIMAX函数报错
这种报错是正常的，我们捕捉错误并记录下来即可 
"""
from math import sqrt
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from warnings import catch_warnings
from warnings import filterwarnings
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
from numpy import array

# 单步Holt Winter指数平滑预测模型
def exp_smoothing_forecast(history, config):
	t,d,s,p,b,r = config
	# 定义模型
	history = array(history)
	model = ExponentialSmoothing(history, trend=t, damped=d, seasonal=s, seasonal_periods=p)
	# 训练模型
	model_fit = model.fit(optimized=True, use_boxcox=b, remove_bias=r)
	# make one step forecast
	yhat = model_fit.predict(len(history), len(history))
	return yhat[0]

# 用标准差作为损失函数
def measure_rmse(actual, predicted):
	return sqrt(mean_squared_error(actual, predicted))

# 将数据按照给定的比例n_test，切分成训练集和测试集
def train_test_split(data, n_test):
	return data[:-n_test], data[-n_test:]

# 时间序列单步预测主函数
def walk_forward_validation(data, n_test, cfg):
    predictions = list()
    # 切分数据
    train, test = train_test_split(data, n_test)
    # seed history with training dataset
    history = [x for x in train]
    history = train
    # 队测试集数据进行单步预测
    for i in range(len(test)):
    	# 调用预测函数，根据观测值进行预测
    	yhat = exp_smoothing_forecast(history, cfg)
    	# 保存预测结果
    	predictions.append(yhat)
    	# 在历史数据中添加一个新观测值，进行下一次预测
    	history.append(test[i])
    # 用标准差评估预测损失
    error = measure_rmse(test, predictions)
    return error

# 对模型参数进行评分，失败的结果返回None
def score_model(data, n_test, cfg, debug=False):
	result = None
	# show all warnings and fail on exception if debugging
	if debug:
		result = walk_forward_validation(data, n_test, cfg)
	else:
		# one failure during model validation suggests an unstable config
		try:
			# never show warnings when grid searching, too noisy
			with catch_warnings():
				filterwarnings("ignore")
				result = walk_forward_validation(data, n_test, cfg)
		except:
			error = None
	# 打印测试的模型参数和对应的准确率
	if result is not None:
		print(' Model[%s] %.3f' % (str(cfg), result))
	return (str(cfg), result)

# 开始网格搜索算法，parallel=False表示调用线程并发计算
def grid_search(data, cfg_list, n_test, parallel=True):
	scores = None
	if parallel:
		# 使用线程并发计算结果
		executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')
		tasks = (delayed(score_model)(data, n_test, cfg) for cfg in cfg_list)
		scores = executor(tasks)
	else:
		scores = [score_model(data, n_test, cfg) for cfg in cfg_list]
	# 把输出结果为None的模型参数删除
	scores = [r for r in scores if r[1] != None]
	# 根据准确率进行排序
	scores.sort(key=lambda tup: tup[1])
	return scores

# 创建一个 指数平滑模型 参数配置列表
# t:trend 趋势，选择趋势性的类型
# d:damped 阻尼，趋势分量是否应该被阻尼
# s:seasonal 季节性，选择季节性的类型
# p:seasonal_periods 季节周期，季节周期数
# b:use_boxcox 是否执行序列的幂变换，或指定变换的lambda
# r:remove_bias 是否通过强制平均残差等于零，消除预测值和拟合值的偏差
def exp_smoothing_configs(seasonal=[None]):
	models = list()
	# define config lists
	t_params = ['add', 'mul', None]
	d_params = [True, False]
	s_params = ['add', 'mul', None]
	p_params = seasonal
	b_params = [True, False]
	r_params = [True, False]
	# create config instances
	for t in t_params:
		for d in d_params:
			for s in s_params:
				for p in p_params:
					for b in b_params:
						for r in r_params:
							cfg = [t,d,s,p,b,r]
							models.append(cfg)
	return models

if __name__ == '__main__':
	# 测试数据集
	data = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
	print(data)
	# 定义数据切分比例
	n_test = 4
	# 指数平滑模型参数配置
	cfg_list = exp_smoothing_configs()
	# 开始网格搜索算法
	scores = grid_search(data, cfg_list, n_test, False)
	print('done')
	# 列出得分排名前三的配置
	for cfg, error in scores[:3]:
		print(cfg, error)
