# univariate lstm example
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

# 构造一元监督学习型数据
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# 获取待预测数据的位置
		end_ix = i + n_steps
		# 如果待预测数据超过序列长度，构造完成
		if end_ix >= len(sequence):
			break
		# 分别汇总 输入 和 输出 数据集
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

# define input sequence
raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
# 定义时间步长，即每组用几个数据作为输入，预测另一个数据
n_steps = 3
# 定义每步包含1个特征值
n_features = 1
# 调用上述split_sequences函数，数据集data(9,3)变成输入输出对：X(6,3),y(6,)
X, y = split_sequence(raw_seq, n_steps)
# 重新调整X的形状，将X(6,3)的输入转换成X(6,3,1)，表示6组数据，每组数据步长为3，每步有1个特征值
X = X.reshape((X.shape[0], X.shape[1], n_features))
# define model
model = Sequential()
# 定义输入的格式input_shape为(None,3,1),因此在fit()时，传入X(6,3,1),y(6,)，模型就会明白这是6组输入输出对
model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
# 密集层,输出(None,1)
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=200, verbose=0)
# demonstrate prediction
x_input = array([70, 80, 90])
x_input = x_input.reshape((1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)