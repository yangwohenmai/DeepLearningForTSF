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

# 单步SARIMA预测函数
# 对于模型来说并不是程序提供的所有参数都是有效的，有的参数传入SARIMAX后，会报错，
# 程序外层会catch这种错误并以None值来做标记，记录下来
def sarima_forecast(history, config):
	order, sorder, trend = config
	# 定义模型
    # order是普通参数，seasonal_order是季节参数，trend是趋势类型
    # 该实现称为SARIMAX而不是SARIMA，因为方法名称的"X"表示该实现还支持外生变量。
	# 外生变量是并行时间序列变量，不是直接通过AR，I或MA流程建模的，而是作为模型的加权输入提供的。
	# 外生变量是可选的，可以通过"exog"参数指定，SARIMAX(data, exog=other_data,...)
	model = SARIMAX(history, order=order, seasonal_order=sorder, trend=trend, enforce_stationarity=False, enforce_invertibility=False)
	# 训练模型过程中会有很多调试信息，disp=0或disp=False表示关闭信息
	model_fit = model.fit(disp=False)
	# 进行预测，有forecast(n)和predict(start,end)两种预测方法，foreast预测是对样本外的数据进行预测，predict可以对样本内和样本外的进行预测：
	# forecast(n)对于输入的训练数据history，每次向后预测n个数值，不写n时默认预测一个值
	# predict(start,end)表示预测从输入训练样本的第一个值开始计数，预测第start到第end个数据。输入5条训练数据，predict(8,9)表示预测第9~10条数据(样本外)，predict(3,6)表示预测第4~7条数据(样本内)
	#yhat = model_fit.forecast()
	#yhat = model_fit.predict(start=len(history),end=len(history))，start和end可以省略
	yhat = model_fit.predict(len(history),len(history))
	# 返回预测数组中的第一条数据
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
			with catch_warnings():
				# 忽略无关的报警信息
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

# 创建一个 SARIMA 参数配置列表,季节周期参数默认为0
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
	# 测试数据集
	data = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
	print(data)
	# 定义数据切分比例
	n_test = 4
	# 模型参数配置
	cfg_list = sarima_configs()
	# 开始网格搜索算法
	scores = grid_search(data, cfg_list, n_test, False)
	print('done')
	# 列出得分排名前三的配置
	for cfg, error in scores[:3]:
		print(cfg, error)
