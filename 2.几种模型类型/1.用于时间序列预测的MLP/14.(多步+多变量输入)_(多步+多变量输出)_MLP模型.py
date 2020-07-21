# multivariate multi-step mlp example
from numpy import array
from numpy import hstack
from keras.models import Sequential
from keras.layers import Dense

# 构造(多步+多变量输入)_(多步+多变量输出)
def split_sequences(sequences, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequences)):
		# 计算输入序列的结尾位置，也是输出序列的起始位置
		end_ix = i + n_steps_in
		# 计算输出序列的结尾位置
		out_end_ix = end_ix + n_steps_out
		# 判断序列是否结束
		if out_end_ix > len(sequences):
			break
		# 根据算好的位置，取输入输出值
		seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, :]
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
# 将3列数据垂直拼接在一起，数据长度要保持一致
dataset = hstack((in_seq1, in_seq2, out_seq))
# n_steps_in:输入数据长度
# n_steps_out:输出数据长度
n_steps_in, n_steps_out = 3, 2
# 构造(多步+多变量输入)_(多步+多变量输出)
X, y = split_sequences(dataset, n_steps_in, n_steps_out)
print(X)
print(y)
# 用(时间步数 * 特征数)来重新调整输入X的形状，将(5,3,3)的输入展平成(5,9)的输入
n_input = X.shape[1] * X.shape[2]
X = X.reshape((X.shape[0], n_input))
# 用(时间步数 * 特征数)来重新调整输出y的形状，将(5,2,3)的输入展平成(5,6)的输入
n_output = y.shape[1] * y.shape[2]
y = y.reshape((y.shape[0], n_output))
# define model
model = Sequential()
model.add(Dense(100, activation='relu', input_dim=n_input))
model.add(Dense(n_output))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=2000, verbose=0)
# demonstrate prediction
x_input = array([[60, 65, 125], [70, 75, 145], [80, 85, 165]])
x_input = x_input.reshape((1, n_input))
yhat = model.predict(x_input, verbose=0)
print(yhat)