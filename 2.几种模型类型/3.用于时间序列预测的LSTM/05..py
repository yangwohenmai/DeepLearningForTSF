from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

# define input sequence
raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
# choose a number of time steps
n_steps = 4
# X(5,4) y(5,)
X, y = split_sequence(raw_seq, n_steps)
# reshape from [samples, timesteps] into [样本数量, 子样本, 时间步, 特征值]
n_features = 1
n_seq = 2
n_steps = 2
"""
[[[[10]
   [20]]
  [[30]
   [40]]]

 [[[20]
   [30]]
  [[40]
   [50]]]

 [[[30]
   [40]]
  [[50]
   [60]]]

 [[[40]
   [50]]
  [[60]
   [70]]]

 [[[50]
   [60]]
  [[70]
   [80]]]]
"""
# X(5,2,2,1)
X = X.reshape((X.shape[0], n_seq, n_steps, n_features))
print(X)
# define model
model = Sequential()
# input_shape=(None, 2, 1)
model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'), input_shape=(None, n_steps, n_features)))
model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=500, verbose=0)
# demonstrate prediction
x_input = array([60, 70, 80, 90])
x_input = x_input.reshape((1, n_seq, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)