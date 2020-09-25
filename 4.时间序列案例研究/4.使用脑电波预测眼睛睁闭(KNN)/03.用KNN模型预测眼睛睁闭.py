from pandas import read_csv
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from numpy import mean
# load the dataset
data = read_csv('EEG_Eye_State_no_outliers.csv', header=None)
values = data.values
scores = list()
# 对数据进行K折交叉验证，K=10，因此会产生10组数据
kfold = KFold(10, shuffle=True, random_state=1)
# 对K折交叉验证构建的10组数据分别建模，并求均值
for train_ix, test_ix in kfold.split(values):
	# 构建测试集和训练集
	trainX, trainy = values[train_ix, :-1], values[train_ix, -1]
	testX, testy = values[test_ix, :-1], values[test_ix, -1]
	# 定义KNN模型，K=3
	model = KNeighborsClassifier(n_neighbors=3)
	# 训练模型并预测
	model.fit(trainX, trainy)
	yhat = model.predict(testX)
	# 评估记录预测值，求均值
	score = accuracy_score(testy, yhat)
	scores.append(score)
	print('>%.3f' % score)
# calculate mean score across each run
print('Final Score: %.3f' % (mean(scores)))