# knn for predicting eye state
from pandas import read_csv
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
# load the dataset
data = read_csv('EEG_Eye_State_no_outliers.csv', header=None)
values = data.values
# split data into inputs and outputs
X, y = values[:, :-1], values[:, -1]
# split the dataset
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.1, shuffle=False, random_state=1)
# define model
model = KNeighborsClassifier(n_neighbors=3)
# fit model on train set
model.fit(trainX, trainy)
# forecast test set
yhat = model.predict(testX)
# evaluate predictions
score = accuracy_score(testy, yhat)
print(score)