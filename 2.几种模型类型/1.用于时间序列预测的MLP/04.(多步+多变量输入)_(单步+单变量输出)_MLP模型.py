# multivariate mlp example
from numpy import array
from numpy import hstack
from keras.models import Sequential
from keras.layers import Dense
"""
本程序构造了一种[10 15 20 25 30 35]->[65]的结构
即：用三组数据的输入来预测一个输出，这种数据本质上具有干扰性，前两组数据的输入和第三组数据的输出无关
但网络仍然能从中学习到正确的计算规则
本文宗旨在于展示如何
"""
# 构造多元监督学习型数据
def split_sequences(sequences, n_steps):
	X, y = list(), list()
	for i in range(len(sequences)):
		# 获取待预测数据的位置
		end_ix = i + n_steps
		# 如果待预测数据超过序列长度，构造完成
		if end_ix > len(sequences)-1:
			break
		# 取前三行数据的前两列作为输入X，第三行数据的最后一列作为输出y
		seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

# 输出: out_seq = in_seq1 + in_seq2
in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])
out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])

"""
重构数据结构：长度为9的一维数组重构成9行1列的结构，(9,)->(9,1)
将[ 25  45  65  85 105 125 145 165 185]
转变成
[[ 25]
 [ 45]
 [ 65]
 [ 85]
 [105]
 [125]
 [145]
 [165]
 [185]]
"""
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
out_seq = out_seq.reshape((len(out_seq), 1))

# 将3列数据垂直拼接在一起，数据长度要保持一致
dataset = hstack((in_seq1, in_seq2, out_seq))
# 设置时间步长度
n_steps = 3
# 构造多元监督学习型数据
X, y = split_sequences(dataset, n_steps)
""" 将输入展平
[[[10 15]
  [20 25]
  [30 35]]

 [[20 25]
  [30 35]
  [40 45]]

 [[30 35]
  [40 45]
  [50 55]]

 [[40 45]
  [50 55]
  [60 65]]

 [[50 55]
  [60 65]
  [70 75]]

 [[60 65]
  [70 75]
  [80 85]]]
转变成
[[10 15 20 25 30 35]
 [20 25 30 35 40 45]
 [30 35 40 45 50 55]
 [40 45 50 55 60 65]
 [50 55 60 65 70 75]
 [60 65 70 75 80 85]]
 """
# 用(时间步数 * 特征数)来重新调整输入X的形状
n_input = X.shape[1] * X.shape[2]
X = X.reshape((X.shape[0], n_input))
# define model
model = Sequential()
model.add(Dense(100, activation='relu', input_dim=n_input))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=2000, verbose=0)
# demonstrate prediction
x_input = array([[80, 85], [90, 95], [100, 105]])
x_input = x_input.reshape((1, n_input))
yhat = model.predict(x_input, verbose=0)
print(yhat)