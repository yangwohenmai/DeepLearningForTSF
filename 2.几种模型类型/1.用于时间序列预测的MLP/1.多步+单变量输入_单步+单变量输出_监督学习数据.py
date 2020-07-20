# univariate data preparation
from numpy import array

# 构造一元监督学习型数据
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
# split into samples
X, y = split_sequence(raw_seq, n_steps)

print(X)
print(y)