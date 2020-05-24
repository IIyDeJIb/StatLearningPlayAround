import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

# crim - per capita crime rate by town
# zn - proportion of residential land zoned for lots over 25,000 sq.ft
# indus - proportion of non-retail business acres per town
# chas - Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
# nox - nitric oxides concentration (parts per 10 million)
# rm - average number of rooms per dwelling
# age - proportion of owner-occupied units built prior to 1940
# dis - weighted distances to five Boston employment centres
# rad - index of accessibility to radial highways
# tax - full-value property-tax rate per USD 10,000
# ptratio - pupil-teacher ratio by town
# black - proportion of blacks by town
# lstat - percentage of lower status of the population
# medv - median value of owner-occupied homes in USD 1000’s

data = pd.read_csv('datasets\\Boston.csv')

# Regress crime rate on all the predictors
fit1 = {}
for pred in data.columns[1:]:
	fit1[pred] = smf.ols('crim ~ eval(pred)', data=data).fit()

# Display p-values
pd.DataFrame([fit1[pred].pvalues[1] for pred in data.columns[1:]],
			 index=data.columns[1:], columns=['p-value'])

# Almost all of the predictors are statistically significant if fit individually

# Multiple regression:
fit2 = smf.ols(
	'crim ~ zn + indus + chas + nox + rm + age + dis + rad + tax+ ptratio + black '
	'+ '
	'lstat + medv',
	data=data).fit()
fit2.summary()

fit1coef = [fit1[pred].params[1] for pred in data.columns[2:]]
ax.scatter(fit1coef, fit2.params[1:])

# Fit a nonlinear model for each predictor
fit1nl = {}
for pred in data.columns[1:]:
	fit1nl[pred] = smf.ols(
		'crim ~ eval(pred) + np.power(eval(pred),2) + np.power(eval(pred),3)',
		data=data).fit()

# Compare the performance of the model based on aic
fit1aic = [fit1[pred].aic for pred in data.columns[2:]]
fit1nlaic = [fit1nl[pred].aic for pred in data.columns[2:]]

# Bar chart to compare the AIC
pd.DataFrame(data={'lin': fit1aic, 'nl': fit1nlaic}, index=data.columns[2:]).plot(
	kind='bar')
# We can see that indeed in some cases the nonlinear model is a better model
pd.plotting.scatter_matrix(data)