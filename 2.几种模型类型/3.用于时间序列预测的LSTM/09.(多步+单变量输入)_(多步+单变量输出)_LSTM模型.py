# univariate multi-step vector-output stacked lstm example
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

# 构建(多步+单变量输入)_(多步+单变量输出)
def split_sequence(sequence, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequence)):
		# 找到输入的结束位置
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out
		# 找到输出的结束位置，输出=输入结束位置+输出长度
		if out_end_ix > len(sequence):
			break
		# 根据计算的位置获取输入输出数据
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

# define input sequence
raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
# 输入数据的步长为3，输出数据的步长为2
n_steps_in, n_steps_out = 3, 2
# 数据集dataset(9,3)变成输入输出对：X(5,3),y(5,2)
X, y = split_sequence(raw_seq, n_steps_in, n_steps_out)
# 将X(5,3)转成X(5,3,1),表示共5组数据，每组3个步长，每个步长1个特征值
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))
# define model
model = Sequential()
# 定义输入的格式input_shape为(3,1),因此在fit()时，传入X(5,3,1),y(5,2)，模型就会明白这是5组输入输出对
model.add(LSTM(30, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
model.add(LSTM(30, activation='relu'))
model.add(Dense(n_steps_out))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=300, verbose=0)

# 构造一条符合要求的输入数据进行测试,将待预测序列x_input(3,)转换成x_input(1,3,1),1表示每批传入1组数据，3表示时间步，1表示特征
x_input = array([60, 70, 80])
x_input = x_input.reshape((1, n_steps_in, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)