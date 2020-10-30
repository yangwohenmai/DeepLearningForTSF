# split data into chunks
from numpy import unique
from pandas import read_csv
from matplotlib import pyplot
import numpy

# 通过'chunkID'列将数据分块
def to_chunks(values, chunk_ix=1):
	chunks = dict()
	# 去重，获取列中包含的序号
	chunk_ids = unique(values[:, chunk_ix])
	# 根据序号获取所有分组
	for chunk_id in chunk_ids:
		# 第一列满足条件的记为True，获取一个True/False列表
		selection = values[:, chunk_ix] == chunk_id
		# 返回标记为True的行
		chunks[chunk_id] = values[selection, :]
	return chunks

# plot distribution of chunk durations
def plot_chunk_durations(chunks):
	# 统计每个数据块内的数据量，k,v是每个chunks.items()的键和值
	chunk_durations = [len(v) for k,v in chunks.items()]
	# boxplot
	pyplot.subplot(2, 1, 1)
	pyplot.boxplot(chunk_durations)
	# histogram
	pyplot.subplot(2, 1, 2)
	pyplot.hist(chunk_durations)
	# histogram
	pyplot.show()

# load dataset
dataset = read_csv(r'D:\咗MyGit\BigDataFile\dsg-hackathon\TrainingData1.csv', header=0)
# group data by chunks
values = dataset.values
chunks = to_chunks(values)
print('Total Chunks: %d' % len(chunks))
# plot chunk durations
plot_chunk_durations(chunks)