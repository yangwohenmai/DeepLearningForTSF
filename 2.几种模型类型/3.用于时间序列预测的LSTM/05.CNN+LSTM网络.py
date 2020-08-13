from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
"""
首先是将输入序列拆分为可以由CNN模型处理的子序列。例如：
先将单变量时间序列数据拆分为输入/输出样本，每个样本有4个时间步长作为输入，1个时间步长作为输出, 即X(None,4) y(None,1)。
然后再将每个样本分为2个子样本，每个子样本具有两个时间步长,一个特征值,即X(None,2,2,1)。CNN可以解析两个时间步长的每个子序列，并将解析结果提供给LSTM模型，以作为输入进行处理。

CNN模型首先用卷积层读取子序列，该卷积层需要指定过滤器和内核大小。过滤器的数量，卷积核的大小、数量是根据经验提前定义好的，而各个卷积核中的权重和偏置是需要通过训练得到的。kernel_size大小是输入序列的每个“读取”操作中包含的时间步数。
卷积层后是最大池化层，该层将过滤器原图精简到其大小的1/2（包括最显著的特征）。然后将这些结构展平为单个一维矢量，作为LSTM层的单个输入的时间步长。

接下来定义LSTM部分，用于解释CNN模型对输入序列的读取并进行预测
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
# X(5,4) y(5,)
X, y = split_sequence(raw_seq, n_steps)
# 将X(5,4) 即[样本数量, 时间步]，重构成X(5,2,2,1) 即[样本数量, 子样本, 时间步, 特征值]
n_features = 1
n_seq = 2
n_steps = 2
"""
[[[[10]
   [20]]
  [[30]
   [40]]]

 [[[20]
   [30]]
  [[40]
   [50]]]

 [[[30]
   [40]]
  [[50]
   [60]]]

 [[[40]
   [50]]
  [[60]
   [70]]]

 [[[50]
   [60]]
  [[70]
   [80]]]]
"""
# X(5,2,2,1)
X = X.reshape((X.shape[0], n_seq, n_steps, n_features))
print(X)
# define model
model = Sequential()
# CNN层的形状64个过滤器，1个卷积核，输入子序列形状input_shape=(None, 2, 1)
model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'), input_shape=(None, n_steps, n_features)))
# 池化层
model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
# 平滑层
model.add(TimeDistributed(Flatten()))
# 后接LSTM层
model.add(LSTM(50, activation='relu'))
# 密集层
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=500, verbose=0)

# 构造一条符合要求的输入数据进行测试
x_input = array([60, 70, 80, 90])
x_input = x_input.reshape((1, n_seq, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)