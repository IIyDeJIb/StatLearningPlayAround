import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import pandas as pd
import statsmodels.api as sm

rnd.seed(1)
x1 = rnd.uniform(size=100)
x2 = 0.5 * x1 + rnd.normal(size=100) / 10
plt.scatter(x1, x2)
np.corrcoef(x1, x2)
y = 2 + 2 * x1 + 0.3 * x2 + rnd.normal(size=100)

fit1 = smf.ols('y ~ x1 + x2', data=pd.DataFrame(data={'y': y, 'x1': x1, 'x2': x2}))

fit1.summary()
# x2 and x1 are collinear, so only one of those comes out statistically significant.


# Add an observation. I will show it affects y~x2 fit.
x1 = np.append(x1,0.1)
x2 = np.append(x2,0.8)
y = np.append(y, 6)

fitx1 = smf.ols('y ~ x1', data=pd.DataFrame(data={'y': y, 'x1': x1, 'x2': x2})).fit()
fitx2 = smf.ols('y ~ x2', data=pd.DataFrame(data={'y': y, 'x1': x1, 'x2': x2})).fit()

fig, ax = plt.subplots()
ax.scatter(x2[:-1],y[:-1])
sm.graphics.abline_plot(intercept=fitx2.params.Intercept, slope=fitx2.params.x2, color='red', ax=ax)
ax.scatter(x2[-1],y[-1], color='magenta')
ax.set_xlim(-1,1)
ax.set_aspect('equal')
sm.graphics.influence_plot(fitx2)