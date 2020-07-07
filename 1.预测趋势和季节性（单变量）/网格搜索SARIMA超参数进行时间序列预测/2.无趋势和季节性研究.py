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
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from pandas import read_csv

# 单步SARIMA预测函数
# 对于模型来说并不是所有参数都是有效的，有的参数传入SARIMAX后，会报错，
# 程序会捕捉这种错误并以None值来做标记，记录下来
def sarima_forecast(history, config):
	order, sorder, trend = config
	# 定义模型
	model = SARIMAX(history, order=order, seasonal_order=sorder, trend=trend, enforce_stationarity=False, enforce_invertibility=False)
	# 训练模型
	model_fit = model.fit(disp=False)
	# 用训练好的模型对历史数据进行预测
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
    # 对测试集数据进行单步预测
    for i in range(len(test)):
    	# 调用预测函数，根据观测值进行预测
    	yhat = sarima_forecast(history, cfg)
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
		except Exception as e:
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

# 创建一个 SARIMA 参数配置列表
def sarima_configs(seasonal=[0]):
	models = list()
	# 定义各种参数的取值范围
	p_params = [0, 1, 2]
	d_params = [0, 1]
	q_params = [0, 1, 2]
	# n,c,t,ct,分别表示无趋势，常数，线性和具有线性趋势的常数
	t_params = ['n','c','t','ct']
	P_params = [0, 1, 2]
	D_params = [0, 1]
	Q_params = [0, 1, 2]
	m_params = seasonal
	# 穷举各种参数配置的组合，记录到列表中
	for p in p_params:
		for d in d_params:
			for q in q_params:
				for t in t_params:
					for P in P_params:
						for D in D_params:
							for Q in Q_params:
								for m in m_params:
									cfg = [(p,d,q), (P,D,Q,m), t]
									models.append(cfg)
	return models

if __name__ == '__main__':
	# load dataset
	series = read_csv('daily-total-female-births.csv', header=0, index_col=0)
	data = series.values
	print(data.shape)
	# data split
	n_test = 165
	# 模型参数配置
	cfg_list = sarima_configs()
	# 开始网格搜索算法
	scores = grid_search(data, cfg_list, n_test, False)
	print('done')
	# 列出得分排名前三的配置
	for cfg, error in scores[:3]:
		print(cfg, error)
