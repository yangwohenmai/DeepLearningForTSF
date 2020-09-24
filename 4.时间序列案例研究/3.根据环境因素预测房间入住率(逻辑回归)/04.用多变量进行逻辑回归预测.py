# 用多变量进行逻辑回归预测
from pandas import read_csv
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
# load the dataset
data = read_csv('combined.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
values = data.values
# 用数据的前n-1列做输入，最后一列作为输出
X, y = values[:, :-1], values[:, -1]
# 分割出30%数据作为训练数据
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.3, shuffle=False, random_state=1)
# 定义逻辑回归预测模型，并进行训练
model = LogisticRegression()
model.fit(trainX, trainy)
# 预测和评估结果
yhat = model.predict(testX)
score = accuracy_score(testy, yhat)
print(score)