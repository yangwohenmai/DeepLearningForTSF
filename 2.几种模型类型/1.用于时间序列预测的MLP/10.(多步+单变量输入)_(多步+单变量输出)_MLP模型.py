# univariate multi-step vector-output mlp example
from numpy import array
from keras.models import Sequential
from keras.layers import Dense

# 构建(多步+单变量输入)_(多步+单变量输出)
def split_sequence(sequence, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequence)):
		# 找到输入的结束位置
		end_ix = i + n_steps_in
		# 找到输出的结束位置，输出=输入结束位置+输出长度
		out_end_ix = end_ix + n_steps_out
		# 判断是否结束
		if out_end_ix > len(sequence):
			break
		# 根据计算的位置获取输入输出数据
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

# define input sequence
raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
# n_steps_in：输入的长度
# n_steps_out：输出的长度 
n_steps_in, n_steps_out = 3, 2

# 构建(多步+单变量输入)_(多步+单变量输出)
X, y = split_sequence(raw_seq, n_steps_in, n_steps_out)
print(X)
print(y)
# define model
model = Sequential()
model.add(Dense(100, activation='relu', input_dim=n_steps_in))
model.add(Dense(n_steps_out))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=2000, verbose=0)
# demonstrate prediction
x_input = array([70, 80, 90])
x_input = x_input.reshape((1, n_steps_in))
yhat = model.predict(x_input, verbose=0)
print(yhat)