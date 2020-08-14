# multivariate cnn example
from numpy import array
from numpy import hstack
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

# 构造多元监督学习型数据
def split_sequences(sequences, n_steps):
	X, y = list(), list()
	for i in range(len(sequences)):
		# 获取待预测数据的位置
		end_ix = i + n_steps
		# 如果待预测数据超过序列长度，构造完成
		if end_ix > len(sequences):
			break
		# 取前三行数据的前两列作为输入X，第三行数据的最后一列作为输出y
		seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
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
# 定义时间步长
n_steps = 3
# 数据集dataset(9,3)变成输入-输出对：X(7,3,2),y(7,)
X, y = split_sequences(dataset, n_steps)
# 定义特征值，直接利用X(7,3,2)中的第3位(特征值)赋值即可
n_features = X.shape[2]
# define model
model = Sequential()
# 定义输入的格式input_shape为(3,2),因此在model.fit时，传入X(7,3,2),y(7,)，模型就会明白这是7组数据
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps, n_features)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
# n_features=1 即预测结果输出位1个值
model.add(Dense(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=1000, verbose=0)

# 将待预测序列x_input(3,2)转换成x_input(1,3,2),1表示每批传入1组数据，3表示时间步，2表示特征
x_input = array([[80, 85], [90, 95], [100, 105]])
x_input = x_input.reshape((1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)