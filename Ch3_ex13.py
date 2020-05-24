import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
import numpy.random as rnd

# Generate the data
np.random.seed(1)
x = rnd.normal(0, 1, 100)
eps = rnd.normal(0, 0.25, 100)
y = -1 + 0.5*x + eps

fit0 = smf.ols('y~x', data=pd.DataFrame({'x': x, 'y': y})).fit()
fit0.summary()

fig, ax = plt.subplots()
ax.scatter(x,y)
sm.graphics.abline_plot(fit0.params[0], fit0.params[1], ax=ax, color='red')
ax.grid()


# Fit the same data with a quadratic term
fit2 = smf.ols('y ~ x + np.square(x)', data=pd.DataFrame({'x': x, 'y': y})).fit()
x_2 = np.linspace(*ax.get_xlim(),num=100)
y_2 = fit2.params.Intercept + fit2.params[1]*x_2 + fit2.params[2]*x_2**2
quad2 = ax.plot(x_2, y_2, color='orange')

# Add legend
# Get the scatter and fit line handles
scath, ablineh, *_ = ax.get_children()
ax.legend([scath, ablineh, *quad2], ['scatter', 'linear fit', 'quadratic fit'])


