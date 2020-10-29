# plot discontiguous chunks
from numpy import nan
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

# plot chunks that do not have all data
def plot_discontiguous_chunks(chunks, row_in_chunk_ix=2):
	n_steps = 8 * 24
	for c_id,rows in chunks.items():
		# skip chunks with all data
		if rows.shape[0] == n_steps:
			continue
		# create empty series
		series = [nan for _ in range(n_steps)]
		# mark all rows with data
		for row in rows:
			# convert to zero offset
			r_id = row[row_in_chunk_ix] - 1
			# mark value
			series[r_id] = c_id
		# plot
		pyplot.plot(series)
	pyplot.show()

# load dataset
dataset = read_csv(r'D:\咗MyGit\BigDataFile\dsg-hackathon\TrainingData1.csv', header=0)
# group data by chunks
values = dataset.values
chunks = to_chunks(values)
# plot discontiguous chunks
plot_discontiguous_chunks(chunks)