import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

data = pd.read_csv('datasets\\Smarket.csv')
pd.plotting.scatter_matrix(data)

fig, ax = plt.subplots()

sns.heatmap(data.corr(), ax=ax)
ax.axis('equal')

# Convert Direction column to numertic
data['Direction'] = data['Direction'].map({'Up': 1, 'Down': 0})

# Two ways to go about the logistic regression in statsmodels - using formula
# approach and no formula
# no formula
logreg_nf = sm.Logit(data['Direction'], sm.add_constant(data.loc[:, 'Lag1':'Volume'])).fit()
logreg_nf.summary()

# from formula
logreg_f = smf.logit('Direction ~ Lag1 + Lag2 + Lag3 + Lag4 + Lag5 + Volume', data=data).fit()
logreg_f.summary()

# The results are identical to the ones in the textbook