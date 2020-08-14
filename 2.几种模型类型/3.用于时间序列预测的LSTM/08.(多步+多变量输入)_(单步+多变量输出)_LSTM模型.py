from numpy import array
from numpy import hstack
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

# 构造多元监督学习型数据
def split_sequences(sequences, n_steps):
	X, y = list(), list()
	for i in range(len(sequences)):
		# 获取待预测数据的位置
		end_ix = i + n_steps
		# 如果待预测数据超过序列长度，构造完成
		if end_ix > len(sequences)-1:
			break
		# 取前三行数据的前两列作为输入X，第三行数据的最后一列作为输出y
		seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]
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
# choose a number of time steps
n_steps = 3
# 输入数据形状X(6,3,3),6组输入数据，每组3个时间步，每个时间步3个特征,y(6,3)，6组数据，每组3个特征值：
X, y = split_sequences(dataset, n_steps)
# 定义特征值，直接利用X(6,3,3)中的第3位(特征值)赋值即可
n_features = X.shape[2]
# define model
model = Sequential()
model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
model.add(LSTM(100, activation='relu'))
model.add(Dense(n_features))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=400, verbose=0)

# 构造一条符合要求的输入数据进行测试,将待预测序列x_input(3,3)转换成x_input(1,3,3),1表示每批传入1组数据，3表示时间步，3表示特征
x_input = array([[70,75,145], [80,85,165], [90,95,185]])
x_input = x_input.reshape((1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)