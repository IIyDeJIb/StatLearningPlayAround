import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import numpy.random as rnd

# Generate the data
np.random.seed(1)
x = rnd.normal(0, 1, 100)
y = 2 * x + rnd.normal(0, 1, 100)
xy = pd.DataFrame(data={'x': x, 'y': y})

# Fit
fit1 = smf.ols('y ~ x + 0', data=xy).fit()
fit1.summary()
fit2 = smf.ols('x ~ y + 0', data=xy).fit()
fit2.summary()

fig, ax = plt.subplots(1, 2)
ax[0].scatter(x, y)
ax[1].scatter(y, x)

print('''The fitted slopes should be the multiplicative inverses of each other. 
This is satisfied approximately: ''', 1 / fit1.params.x * 1 / fit2.params.y)

# Compare the t values for x~y and y~x
print('t-value for y~x: ', fit1.tvalues)
print('t-value for x~y: ', fit2.tvalues)
print('The values are equal.')
print()
print('This also holds for linear regression with an intercept.')

# Generate an example where x~y and y~x is the same (or x = y)
x = 100 * rnd.random(100)
y = x

fit_eq = smf.ols('x~y', data=pd.DataFrame({'x': x, 'y': y})).fit()

