# spot check for ES1
from numpy import mean
from numpy import std
from pandas import read_csv
from matplotlib import pyplot
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 加载维度为25的数据集
dataset = read_csv('25_es2_train.csv', header=None)
# 将每条数据的输入输出分离出来
values = dataset.values
X, y = values[:, :-1], values[:, -1]

# 对KNN的K值从1到21进行网格搜索，找到最优值
all_scores, names = list(), list()
for k in range(1,22):
	# evaluate
	scaler = StandardScaler()
	model = KNeighborsClassifier(n_neighbors=k)
	pipeline = Pipeline(steps=[('s',scaler), ('m',model)])
	names.append(str(k))
	scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=5, n_jobs=-1)
	all_scores.append(scores)
	# summarize
	m, s = mean(scores)*100, std(scores)*100
	print('k=%d %.3f%% +/-%.3f' % (k, m, s))
# plot
pyplot.boxplot(all_scores, labels=names)
#pyplot.show()