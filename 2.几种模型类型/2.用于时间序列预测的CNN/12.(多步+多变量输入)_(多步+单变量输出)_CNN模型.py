# multivariate multi-step 1d cnn example
from numpy import array
from numpy import hstack
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

# 构造(多步+多变量输入)_(多步+单变量输出)
def split_sequences(sequences, n_steps_in, n_steps_out, shift=0):
	X, y = list(), list()
	for i in range(len(sequences)):
		# 计算输入序列的结尾位置
		end_ix = i + n_steps_in
		# 计算输出序列的起始位置
		out_start_ix = end_ix + shift
		# 计算输出序列的结尾位置
		out_end_ix = end_ix + n_steps_out + shift
		# 判断序列是否结束
		if out_end_ix > len(sequences):
			break
		# 根据算好的位置，取输入输出值
		seq_x, seq_y = sequences[i:end_ix, :-1], sequences[out_start_ix:out_end_ix, -1]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

# define input sequence
in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])
out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])
# 重构数据结构：长度为9的一维数组重构成9行1列的二维数组，(9,)->(9,1)
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
out_seq = out_seq.reshape((len(out_seq), 1))
# horizontally stack columns
dataset = hstack((in_seq1, in_seq2, out_seq))
# n_steps_in:输入数据长度
# n_steps_out:输出数据长度
# shift:从距离输入序列结尾位置，上下偏移多少位开始取值
n_steps_in, n_steps_out, shift = 3, 2, 0
# 调用上述split_sequences函数，数据集data(9,3)变成输入输出对：X(5,3,2),y(5,2)
X, y = split_sequences(dataset, n_steps_in, n_steps_out, shift)
# 定义特征值，直接利用X(5,3,2)中的第3位(特征值)赋值即可
n_features = X.shape[2]
# define model
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps_in, n_features)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(n_steps_out))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=2000, verbose=0)

# 构造一条符合要求的输入数据进行测试,将待预测序列x_input(3,)转换成x_input(1,3,2),1表示每批传入1组数据，3表示时间步，2表示特征
x_input = array([[70, 75], [80, 85], [90, 95]])
x_input = x_input.reshape((1, n_steps_in, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)