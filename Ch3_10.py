import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

csData = pd.read_csv('datasets\\Carseats.csv')

csData['Urban'] = csData['Urban'].astype('category')
csData['US'] = csData['US'].astype('category')
fit_d = smf.ols('Sales ~ Price + Urban + US', data=csData).fit()
fit_d.summary()

fit_e = smf.ols('Sales ~ US + Price', data=csData).fit()

# Compare the fits
print(' fit (e) AIC: ', fit_e.aic)
print(' fit (d) AIC: ', fit_d.aic)

# Examine the ouliers and the high-leverage observations
sm.graphics.influence_plot(fit_e)