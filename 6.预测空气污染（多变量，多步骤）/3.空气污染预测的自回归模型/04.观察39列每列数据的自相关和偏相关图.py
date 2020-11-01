# acf and pacf plots
from numpy import loadtxt
from numpy import nan
from numpy import isnan
from numpy import count_nonzero
from numpy import unique
from numpy import nanmedian
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

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

# 使用所有其他数据对应同一时间的中值作为估算值进行填充
def impute_missing(rows, hours, series, col_ix):
	# count missing observations
	n_missing = count_nonzero(isnan(series))
	# calculate ratio of missing
	ratio = n_missing / float(len(series)) * 100
	# 如果是100，说明没有数据缺失
	if ratio == 100.0:
		return series
	# impute missing using the median value for hour in the series
	imputed = list()
	for i in range(len(series)):
		if isnan(series[i]):
			# get all rows with the same hour
			matches = rows[rows[:,2]==hours[i]]
			# fill with median value
			value = nanmedian(matches[:, col_ix])
			imputed.append(value)
		else:
			imputed.append(series[i])
	return imputed

# 展示训练集包含的前120条数据的缺失情况
def variable_to_series(chunk_train, col_ix, n_steps=5*24):
	# lay out whole series
	data = [nan for _ in range(n_steps)]
	# mark all available data
	for i in range(len(chunk_train)):
		# get position in chunk
		position = int(chunk_train[i, 1] - 1)
		# store data
		data[position] = chunk_train[i, col_ix]
	return data

# plot acf and pacf plots for each imputed variable series
def plot_variables(chunk_train, hours, n_vars=39):
	pyplot.figure()
	n_plots = n_vars * 2
	j = 0
	lags = 24
	for i in range(1, n_plots, 2):
		# convert target number into column number
		col_ix = 3 + j
		j += 1
		# get series
		series = variable_to_series(chunk_train, col_ix)
		imputed = impute_missing(chunk_train, hours, series, col_ix)
		# acf
		axis = pyplot.subplot(n_vars, 2, i)
		plot_acf(imputed, ax=axis, lags=lags, zero=False)
		axis.set_title('')
		axis.set_xticklabels([])
		axis.set_yticklabels([])
		# pacf
		axis = pyplot.subplot(n_vars, 2, i+1)
		plot_pacf(imputed, ax=axis, lags=lags, zero=False)
		axis.set_title('')
		axis.set_xticklabels([])
		axis.set_yticklabels([])
	# show plot
	pyplot.show()


train = loadtxt(r'D:\咗MyGit\BigDataFile\dsg-hackathon\naive_train.csv', delimiter=',')
# group data by chunks
train_chunks = to_chunks(train)
# pick one chunk
rows = train_chunks[0]
# prepare sequence of hours for the chunk
hours = variable_to_series(rows, 2)
# plot variables
plot_variables(rows, hours)