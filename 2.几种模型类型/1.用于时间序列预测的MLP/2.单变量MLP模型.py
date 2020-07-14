# univariate mlp example
from numpy import array
from keras.models import Sequential
from keras.layers import Dense

# 构造监督学习型数据
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# 获取待预测数据的位置
		end_ix = i + n_steps
		# 如果待预测数据超过序列长度，构造完成
		if end_ix > len(sequence)-1:
			break
		# 分别汇总 输入 和 输出 数据集
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

# define input sequence
raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
# 定义时间步，即用几个数据作为输入，预测另一个数据
n_steps = 3
# 构造监督学习型数据
X, y = split_sequence(raw_seq, n_steps)
# 定义一个简单的MLP模型
model = Sequential()
model.add(Dense(100, activation='relu', input_dim=n_steps))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=2000, verbose=0)
# 手动定义一个输入
x_input = array([70, 80, 90])
# 将(3,)的序列[70, 80, 90]，重构成(1, 3)的序列[[70, 80, 90]]
x_input = x_input.reshape((1, n_steps))
yhat = model.predict(x_input, verbose=0)
print(yhat)