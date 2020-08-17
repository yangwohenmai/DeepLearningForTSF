from numpy import array
from numpy import hstack
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed

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
# 调用上述split_sequences函数，数据集data(9,3)变成输入输出对：X(5,3,3),y(5,2,3)
X, y = split_sequences(dataset, n_steps_in, n_steps_out)
# 取X(5,3,3)的特征值3作为输入特征值
n_features = X.shape[2]
# define model
model = Sequential()
# 定义输入的格式input_shape为(3,3),因此在fit()时，传入X(5,3,3),y(5,2,3)，模型就会明白这是5组输入输出对
model.add(LSTM(200, activation='relu', input_shape=(n_steps_in, n_features)))
model.add(RepeatVector(n_steps_out))
model.add(LSTM(200, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(n_features)))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=300, verbose=0)

# 构造一条符合要求的输入数据进行测试,将待预测序列x_input(3,3)转换成x_input(1,3,3),1表示每批传入1组数据，3表示时间步，3表示特征
x_input = array([[60, 65, 125], [70, 75, 145], [80, 85, 165]])
x_input = x_input.reshape((1, n_steps_in, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)