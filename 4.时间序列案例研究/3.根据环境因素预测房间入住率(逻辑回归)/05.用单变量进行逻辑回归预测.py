# 用单变量进行逻辑回归预测
from pandas import read_csv
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
# load the dataset
data = read_csv('combined.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
values = data.values
# 数据文件中5列分别代表5种变量
features = [0, 1, 2, 3, 4]
# 分别用五种单变量单独进行逻辑回归
for f in features:
	# 从数据中提取出每一列分别作为单变量输入和输出，
	X, y = values[:, f].reshape((len(values), 1)), values[:, -1]
	# 数据中30%为测试集
	trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.3, shuffle=False, random_state=1)
	# 定义和训练逻辑回归模型
	model = LogisticRegression()
	model.fit(trainX, trainy)
	# 预测
	yhat = model.predict(testX)
	# 评估模型
	score = accuracy_score(testy, yhat)
	print('feature=%d, name=%s, score=%.3f' % (f, data.columns[f], score))