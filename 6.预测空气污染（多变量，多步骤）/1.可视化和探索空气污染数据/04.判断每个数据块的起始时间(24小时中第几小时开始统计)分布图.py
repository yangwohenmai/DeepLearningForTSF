# plot distribution of chunk start hour
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

# plot distribution of chunk start hour
def plot_chunk_start_hour(chunks, hour_in_chunk_ix=5):
	# chunk start hour
	chunk_start_hours = [v[0, hour_in_chunk_ix] for k,v in chunks.items() if len(v)==192]
	# boxplot
	pyplot.subplot(2, 1, 1)
	pyplot.boxplot(chunk_start_hours)
	# histogram
	pyplot.subplot(2, 1, 2)
	pyplot.hist(chunk_start_hours, bins=24)
	# histogram
	pyplot.show()

# load dataset
dataset = read_csv(r'D:\咗MyGit\BigDataFile\dsg-hackathon\TrainingData1.csv', header=0)
# group data by chunks
values = dataset.values
chunks = to_chunks(values)
# plot distribution of chunk start hour
plot_chunk_start_hour(chunks)