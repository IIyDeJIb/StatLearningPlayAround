# Stat Learning text; Chapter 3, problem 8
import pandas as pd
import statsmodels.formula.api as smf
from abline import abline
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

# read the csv
auto = pd.read_csv('datasets\\auto.csv', usecols=['mpg', 'horsepower', 'name'])
auto = auto.set_index('name')

# to_numeric is useful in that anything which doesn't fit the numeric template
# will be set to nan if error=='coerce'
auto['horsepower'] = pd.to_numeric(auto['horsepower'], errors='coerce')

# Fit the model
fit1 = smf.ols('mpg ~ horsepower', data=auto).fit()
slope1 = fit1.params.horsepower
intercept1 = fit1.params.Intercept

# a bit weird, but the prediction method needs an exog matrix which contains all
# the variables.
exog1 = pd.DataFrame({'Intercept': 1, 'horsepower': 98}, index=[0])
pred1 = fit1.get_prediction(exog1)

# Predicted mpg
val1 = pred1.predicted_mean

# Prediction interval (goes by observation confidence interval in statsmodels)
predInt1 = pred1.summary_frame()[['obs_ci_lower', 'obs_ci_upper']]

# Confidence interval
conf1 = pred1.summary_frame()[['mean_ci_lower', 'mean_ci_upper']]

# Alternatively
# conf1 = pred1.conf_int()

# (b) Plot the response and the predictor.
import statsmodels.api as sm
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2)

# Plot the data
auto.plot(x='horsepower', y='mpg', style='o', ax=ax[0])

# Plot the regression line
sm.graphics.abline_plot(intercept=intercept1, slope=slope1, ax=ax[0], color='red')
ax[0].set_title('Scatter plot')
ax[0].set_xlabel('horsepower')
ax[0].set_ylabel('mpg')

# Plot residuals
ax[1].plot(fit1.resid.values, linestyle='--', marker='o')
sm.graphics.abline_plot(0, 0, ax=ax[1], color='red', linestyle='--')
ax[1].set_title('Residuals')

# Plot studentized residuals vs leverage (influence plot)
# From the plot we can see that even though there are high leverage and outlying
# observations, those are not separate and there is a trail of observations
# leading to it. This suggests that the problem is not in the observations but in
# the model not taking into account the non-linearity.
sm.graphics.influence_plot(fit1)

# --------- 9 -----------
# Scatter matrix
pd.plotting.scatter_matrix(auto1)

# b. Calculate and visualize correlation between the columns
fig, ax = plt.subplots()
plt.imshow(auto1.corr().values)

# c.
fit2 = fit2 = smf.ols(
	'mpg ~ cylinders + displacement + horsepower + weight + acceleration + year + '
	'origin',
	data=auto1).fit()

fit2.summary()

# d. find the high-leverage Buick observation
sm.graphics.influence_plot(fit2)

# e. Interaction terms
smf.ols(
	'mpg ~ cylinders + displacement + horsepower + weight + '
	'acceleration*horsepower '
	'+ year',
	data=auto1).fit().summary()

# f. data transformation
fig, ax = plt.subplots(2, 2)
fig1, ax1 = plt.subplots(2, 2)
ax[0, 0].scatter(x=auto['horsepower'], y=auto['mpg'])
ax[0, 1].scatter(x=np.log(auto['horsepower']), y=auto['mpg'])
ax[1, 0].scatter(x=np.square(auto['horsepower']), y=auto['mpg'])
ax[1, 1].scatter(x=np.sqrt(auto['horsepower']), y=auto['mpg'])
ax[0, 1].set_title('log')
ax[0, 0].set_title('original')
ax[1, 1].set_title('sqrt')
ax[1, 0].set_title('square')

fit_trans = []
fit_trans.append(smf.ols("mpg ~ horsepower", data=auto).fit())
fit_trans.append(smf.ols("mpg ~ np.log(horsepower)", data=auto).fit())
fit_trans.append(smf.ols("mpg ~ np.exp(horsepower)", data=auto).fit())
fit_trans.append(smf.ols("mpg ~ np.sqrt(horsepower)", data=auto).fit())

for ii in range(4):
	sm.graphics.abline_plot(slope=fit_trans[ii].params[1],
							intercept=fit_trans[ii].params[0],
							ax=ax.flatten()[ii],
							color='red')
	ax1.flatten()[ii].plot(fit_trans[ii].resid.values)
	abline(0, 0, ax=ax1.flatten()[ii])
	ax1[0, 1].set_title('log')
	ax1[0, 0].set_title('original')
	ax1[1, 1].set_title('sqrt')
	ax1[1, 0].set_title('exp')




