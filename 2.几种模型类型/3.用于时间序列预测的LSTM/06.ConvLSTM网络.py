from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import ConvLSTM2D
"""
ConvLSTM中卷积读取直接内置到每个LSTM单元中。
ConvLSTM是为读取二维数据而开发的，但也适用于单变量时间序列预测。该层期望输入为二维图像序列，因此输入数据的形状必须为：
[samples, timesteps, rows, columns, features]->[样本数量, 时间步长, 行, 列, 特征值]
"""
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
# split into samples
X, y = split_sequence(raw_seq, n_steps)
# 定义序列数n_seq，时间步数time_steps，特征值n_features
n_features = 1
n_seq = 2
time_steps = 2
# 将X(5,4) 即[样本数量, 时间步]，重构成X(5,2,1,2,1) 即[样本数量, 时间步长, 行, 列, 特征值]
# 将每个样本分成多个子序列，其中“时间步长”对应的是子序列数（即n_seq），“列”对应的是每个子序列的时间步数（即time_steps），在处理一维数据时，“行”固定为1
X = X.reshape((X.shape[0], n_seq, 1, time_steps, n_features))

model = Sequential()
# 定义单层ConvLSTM，kernel_size=(行，列)，处理一维序列时，内核中的行数始终固定为1,定义输入数据形状input_shape=(2,1,2,1)
model.add(ConvLSTM2D(filters=64, kernel_size=(1,2), activation='relu', input_shape=(n_seq, 1, time_steps, n_features)))
model.add(Flatten())
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=500, verbose=0)

# 构造一条符合要求的输入数据进行测试
x_input = array([60, 70, 80, 90])
x_input = x_input.reshape((1, n_seq, 1, time_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)