# autoregression forecast
from numpy import loadtxt
from numpy import nan
from numpy import isnan
from numpy import count_nonzero
from numpy import unique
from numpy import array
from numpy import nanmedian
from statsmodels.tsa.arima_model import ARIMA
from matplotlib import pyplot
from warnings import catch_warnings
from warnings import filterwarnings

# 将数据按照'chunkID', 分块
def to_chunks(values, chunk_ix=0):
	chunks = list()
	# get the unique chunk ids
	chunk_ids = unique(values[:, chunk_ix])
	# group rows by chunk id
	for chunk_id in chunk_ids:
		selection = values[:, chunk_ix] == chunk_id
		chunks.append(values[selection, :])
	return chunks

# 要预测的时间点
def get_lead_times():
	return [1, 2, 3, 4, 5, 10, 17, 24, 48, 72]

# 如果数据并非全为nan，返回true
def has_data(data):
	return count_nonzero(isnan(data)) < len(data)

# 用当前块中相同小时的中位数为缺失数据进行插值
def impute_missing(rows, hours, series, col_ix):
	imputed = list()
	for i in range(len(series)):
		if isnan(series[i]):
			# 获取当前时间块中所有相同小时的数据
			matches = rows[rows[:,2]==hours[i]]
			# fill with median value
			value = nanmedian(matches[:, col_ix])
			if isnan(value):
				value = 0.0
			imputed.append(value)
		else:
			imputed.append(series[i])
	return imputed

# 给数据块中有时间缺失的位置赋nan值
def variable_to_series(chunk_train, col_ix, n_steps=5*24):
	# 创建一个全nan的data列表
	data = [nan for _ in range(n_steps)]
	# 给data列表中索引对应的位置赋值
	for i in range(len(chunk_train)):
		# 获取数据索引
		position = int(chunk_train[i, 1] - 1)
		# 在索引位置存入数据
		data[position] = chunk_train[i, col_ix]
	return data

# fit ARIMA model and generate a forecast
def fit_and_forecast(series):
	# define the model
	model = ARIMA(series, order=(1,0,0))
	# return a nan forecast in case of exception
	try:
		# ignore statsmodels warnings
		with catch_warnings():
			filterwarnings("ignore")
			# fit the model
			model_fit = model.fit(disp=False)
			# forecast 72 hours
			yhat = model_fit.predict(len(series), len(series)+72)
			# extract lead times
			lead_times = array(get_lead_times())
			indices = lead_times - 1
			return yhat[indices]
	except:
		return [nan for _ in range(len(get_lead_times()))]

# 整理数据、训练模型、预测数据
def forecast_variable(hours, chunk_train, chunk_test, lead_times, target_ix):
	# 将目标列转换成表格中对应的列数
	col_ix = 3 + target_ix
	# 如果数据块中没有数据，直接返回nan
	if not has_data(chunk_train[:, col_ix]):
		forecast = [nan for _ in range(len(lead_times))]
		return forecast
	# 获取在缺失值处补nan后的时间列表
	series = variable_to_series(chunk_train, col_ix)
	# 用中位数对缺失数据进行插值
	imputed = impute_missing(chunk_train, hours, series, col_ix)
	# 训练ARIMA模型并与进行预测
	forecast = fit_and_forecast(imputed)
	return forecast

# 将预测好的数据转换成 [chunk][variable][time] 格式
def forecast_chunks(train_chunks, test_input):
	lead_times = get_lead_times()
	predictions = list()
	# enumerate chunks to forecast
	for i in range(len(train_chunks)):
		# prepare sequence of hours for the chunk
		hours = variable_to_series(train_chunks[i], 2)
		# enumerate targets for chunk
		chunk_predictions = list()
		for j in range(39):
			yhat = forecast_variable(hours, train_chunks[i], test_input[i], lead_times, j)
			chunk_predictions.append(yhat)
		chunk_predictions = array(chunk_predictions)
		predictions.append(chunk_predictions)
	return array(predictions)

# 将测试集真实数据转换成 [chunk][variable][time] 格式
def prepare_test_forecasts(test_chunks):
	predictions = list()
	# enumerate chunks to forecast
	for rows in test_chunks:
		# enumerate targets for chunk
		chunk_predictions = list()
		for j in range(3, rows.shape[1]):
			yhat = rows[:, j]
			chunk_predictions.append(yhat)
		chunk_predictions = array(chunk_predictions)
		predictions.append(chunk_predictions)
	return array(predictions)

# 计算预测数据的误差值
def calculate_error(actual, predicted):
	# give the full actual value if predicted is nan
	if isnan(predicted):
		return abs(actual)
	# calculate abs difference
	return abs(actual - predicted)

# evaluate a forecast in the format [chunk][variable][time]
def evaluate_forecasts(predictions, testset):
	lead_times = get_lead_times()
	total_mae, times_mae = 0.0, [0.0 for _ in range(len(lead_times))]
	total_c, times_c = 0, [0 for _ in range(len(lead_times))]
	# enumerate test chunks
	for i in range(len(test_chunks)):
		# convert to forecasts
		actual = testset[i]
		predicted = predictions[i]
		# enumerate target variables
		for j in range(predicted.shape[0]):
			# enumerate lead times
			for k in range(len(lead_times)):
				# skip if actual in nan
				if isnan(actual[j, k]):
					continue
				# calculate error
				error = calculate_error(actual[j, k], predicted[j, k])
				# update statistics
				total_mae += error
				times_mae[k] += error
				total_c += 1
				times_c[k] += 1
	# normalize summed absolute errors
	total_mae /= total_c
	times_mae = [times_mae[i]/times_c[i] for i in range(len(times_mae))]
	return total_mae, times_mae

# 统计模型得分
def summarize_error(name, total_mae, times_mae):
	# print summary
	lead_times = get_lead_times()
	formatted = ['+%d %.3f' % (lead_times[i], times_mae[i]) for i in range(len(lead_times))]
	s_scores = ', '.join(formatted)
	print('%s: [%.3f MAE] %s' % (name, total_mae, s_scores))
	# plot summary
	pyplot.plot([str(x) for x in lead_times], times_mae, marker='.')
	pyplot.show()


train = loadtxt(r'D:\咗MyGit\BigDataFile\dsg-hackathon\naive_train.csv', delimiter=',')
test = loadtxt(r'D:\咗MyGit\BigDataFile\dsg-hackathon\naive_test.csv', delimiter=',')
# 数据分块
train_chunks = to_chunks(train)
test_chunks = to_chunks(test)
# forecast
test_input = [rows[:, :3] for rows in test_chunks]
# 整理数据、训练模型、预测数据，最后将预测数据转换成[chunk][variable][time]格式
forecast = forecast_chunks(train_chunks, test_input)
# 将测试集真实数据转换成 [chunk][variable][time] 格式
actual = prepare_test_forecasts(test_chunks)
# 评估预测数据的误差
total_mae, times_mae = evaluate_forecasts(forecast, actual)
# 统计模型得分
summarize_error('ARIMA', total_mae, times_mae)