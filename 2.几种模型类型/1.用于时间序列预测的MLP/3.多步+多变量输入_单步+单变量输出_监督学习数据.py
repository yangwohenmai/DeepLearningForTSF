# multivariate data preparation
from numpy import array
from numpy import hstack

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

# 输出： out_seq = in_seq1 + in_seq2
in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])
out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])
# 重构数据结构：(9,)->(9,1)
# [10 20 30 40 50 60 70 80 90]->[[10] [20] [30] [40] [50] [60] [70] [80] [90]]
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
out_seq = out_seq.reshape((len(out_seq), 1))
# 将3列数据垂直拼接在一起，数据长度要保持一致
dataset = hstack((in_seq1, in_seq2, out_seq))
# 设置时间步长度
n_steps = 3
# 构造多元监督学习型数据
X, y = split_sequences(dataset, n_steps)
print(X.shape, y.shape)

print(X)
print(y)