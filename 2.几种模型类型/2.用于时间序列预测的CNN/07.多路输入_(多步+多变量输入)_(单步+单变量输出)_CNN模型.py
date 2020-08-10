# multivariate multi-headed 1d cnn example
from numpy import array
from numpy import hstack
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.merge import concatenate

# split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the dataset
		if end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

# define input sequence
in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])
out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])
# convert to [rows, columns] structure
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
out_seq = out_seq.reshape((len(out_seq), 1))
# horizontally stack columns
dataset = hstack((in_seq1, in_seq2, out_seq))
# choose a number of time steps
n_steps = 3
# convert into input/output
X, y = split_sequences(dataset, n_steps)
# one time series per head
n_features = 1
"""
将以下数据（7,3,2）：
[[[10 15]
  [20 25]
  [30 35]]
 [[20 25]
  [30 35]
  [40 45]]
 [[30 35]
  [40 45]
  [50 55]]
 [[40 45]
  [50 55]
  [60 65]]
 [[50 55]
  [60 65]
  [70 75]]
 [[60 65]
  [70 75]
  [80 85]]
 [[70 75]
  [80 85]
  [90 95]]]

拆分成两段输入数据(7,3,1)：
[[[10]
  [20]
  [30]]
 [[20]
  [30]
  [40]]
 [[30]
  [40]
  [50]]
 [[40]
  [50]
  [60]]
 [[50]
  [60]
  [70]]
 [[60]
  [70]
  [80]]
 [[70]
  [80]
  [90]]]
  和
[[[15]
  [25]
  [35]]
 [[25]
  [35]
  [45]]
 [[35]
  [45]
  [55]]
 [[45]
  [55]
  [65]]
 [[55]
  [65]
  [75]]
 [[65]
  [75]
  [85]]
 [[75]
  [85]
  [95]]]
"""
X1 = X[:, :, 0].reshape(X.shape[0], X.shape[1], n_features)
X2 = X[:, :, 1].reshape(X.shape[0], X.shape[1], n_features)
# 分别定义两路输入的输入数据形状，（None,3,1），第一个None元素代表有N组数据
visible1 = Input(shape=(n_steps, n_features))
# 卷积层(,3,1)->(,2,64)
cnn1 = Conv1D(filters=64, kernel_size=2, activation='relu')(visible1)
# 池化层(,2,64)->(,1,64)
cnn1 = MaxPooling1D(pool_size=2)(cnn1)
# 平滑层(,1,64)->(,64)
cnn1 = Flatten()(cnn1)
# 分别定义两路输入的输入数据形状，（None,3,1），第一个None元素代表有N组数据
visible2 = Input(shape=(n_steps, n_features))
# 卷积层(None,3,1)->(None,2,64)
cnn2 = Conv1D(filters=64, kernel_size=2, activation='relu')(visible2)
# 池化层(None,2,64)->(None,1,64)
cnn2 = MaxPooling1D(pool_size=2)(cnn2)
# 平滑层(None,1,64)->(None,64)
cnn2 = Flatten()(cnn2)
# 两路数据通过卷积层，池化层，平滑层后，合并起来，与全连接层相连[(None,64),(None,64)]->(None,128)
merge = concatenate([cnn1, cnn2])
# 全连接层(None,128)->(None,50)
dense = Dense(50, activation='relu')(merge)
# 全连接层(None,50)->(None,1)
output = Dense(1)(dense)
model = Model(inputs=[visible1, visible2], outputs=output)
model.compile(optimizer='adam', loss='mse')
# 训练网络时，输入数据拼接成[X1, X2]即[(7,3,1),(7,3,1)]
model.fit([X1, X2], y, epochs=1000, verbose=0)

# 构造输入数据
"""
[[ 80  85]
 [ 90  95]
 [100 105]]
"""
x_input = array([[80, 85], [90, 95], [100, 105]])
"""
[[[ 80]
  [ 90]
  [100]]]
"""
x1 = x_input[:, 0].reshape((1, n_steps, n_features))
"""
[[[ 85]
  [ 95]
  [105]]]
"""
x2 = x_input[:, 1].reshape((1, n_steps, n_features))
yhat = model.predict([x1, x2], verbose=0)
print(yhat)