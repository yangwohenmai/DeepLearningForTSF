from pandas import read_csv
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_pacf
series = read_csv('daily-min-temperatures.csv', header=0, index_col=0)
plot_pacf(series, lags=50)
pyplot.show()