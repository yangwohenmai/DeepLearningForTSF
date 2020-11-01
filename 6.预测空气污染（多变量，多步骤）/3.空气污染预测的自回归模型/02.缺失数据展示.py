# plot missing
from numpy import loadtxt
from numpy import nan
from numpy import unique
from matplotlib import pyplot

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

# 39个字段每个都进行展示
def plot_variables(chunk_train, n_vars=39):
	pyplot.figure()
	for i in range(n_vars):
		# convert target number into column number
		col_ix = 3 + i
		# mark missing obs for variable
		series = variable_to_series(chunk_train, col_ix)
		# plot
		ax = pyplot.subplot(n_vars, 1, i+1)
		ax.set_xticklabels([])
		ax.set_yticklabels([])
		pyplot.plot(series)
	# show plot
	pyplot.show()


train = loadtxt(r'D:\咗MyGit\BigDataFile\dsg-hackathon\naive_train.csv', delimiter=',')
# group data by chunks
train_chunks = to_chunks(train)
# pick one chunk
rows = train_chunks[0]
# plot variables
plot_variables(rows)