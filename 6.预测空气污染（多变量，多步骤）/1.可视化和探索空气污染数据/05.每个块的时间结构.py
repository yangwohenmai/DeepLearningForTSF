# plot inputs for a chunk
from numpy import unique
from pandas import read_csv
from matplotlib import pyplot

# split the dataset by 'chunkID', return a dict of id to rows
def to_chunks(values, chunk_ix=1):
	chunks = dict()
	# get the unique chunk ids
	chunk_ids = unique(values[:, chunk_ix])
	# group rows by chunk id
	for chunk_id in chunk_ids:
		selection = values[:, chunk_ix] == chunk_id
		chunks[chunk_id] = values[selection, :]
	return chunks

# plot all inputs for one or more chunk ids
def plot_chunk_inputs(chunks, c_ids):
	pyplot.figure()
	inputs = range(6, 56)
	for i in range(len(inputs)):
		ax = pyplot.subplot(len(inputs), 1, i+1)
		ax.set_xticklabels([])
		ax.set_yticklabels([])
		column = inputs[i]
		for chunk_id in c_ids:
			rows = chunks[chunk_id]
			pyplot.plot(rows[:,column])
	pyplot.show()

# load data
dataset = read_csv(r'D:\咗MyGit\BigDataFile\dsg-hackathon\TrainingData1.csv', header=0)
# group data by chunks
values = dataset.values
chunks = to_chunks(values)
# plot inputs for some chunks
plot_chunk_inputs(chunks, [1])