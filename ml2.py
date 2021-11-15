import pandas as pd
from sklearn.model_selection import train_test_split

nyc = pd.read_csv("ave_hi_nyc_jan_1895-2018.csv")

print(nyc.head(3))

# returns a numpy array containing the Date column's values
print(nyc.Date.values)

# reshape(-1,1) tells reshape to infer the number of rows based on the number of columns (1)and the number of elements (124) in the array
print(nyc.Date.values.reshape(-1,1))

# split the data for training and testing
x_train, x_test, y_train, y_test = train_test_split(nyc.Date.values.reshape(-1,1), nyc.Temperature.values, random_state = 11)
print(x_train.shape)
print(x_test.shape)


from sklearn.linear_model import LinearRegression
lr = LinearRegression()

# the fit method expects the samples and the targets for training
lr.fit(X = x_train, y = y_train)

coef = lr.coef_ # m in the slope equation
intercept = lr.intercept_ # b in the slope equation

# testing the model
predicted = lr.predict(x_test)
expected = y_test

# check the accuracy
print(predicted[:20])
print(expected[:20])

# predicting future and estimating past temperatures 
# lambda implements y = mx + b
predict = lambda x: coef * x + intercept
print(predict(2025))


# visualizing the dataset with a regression line

import seaborn as sns

axes = sns.scatterplot(data = nyc, x = "Date", y = "Temperature", hue = "Temperature", palette = "winter", legend = False)
axes.set_ylim(10, 70) 

import numpy as np

x = np.array([min(nyc.Date.values), max(nyc.Date.values)])
print(x)
y = predict(x)
print(y)

import matplotlib.pyplot as plt

line = plt.plot(x, y)

plt.show()
