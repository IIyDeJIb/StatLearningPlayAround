import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

data = pd.read_csv('datasets\\Weekly.csv')

data.info()

sns.heatmap(data.corr())

pd.plotting.scatter_matrix(data)

# Logistic regression by sklearn
X = data.loc[:, 'Lag1':'Volume']
y = data['Direction']
data['Direction'] = data['Direction'].map({'Up': 1, 'Down': 0})
logref_sk = LogisticRegression().fit(X, y)
y_pred = logref_sk.predict(X)
confusion_matrix(y, y_pred)

# Fit with statsmodels to see R-like summary
logreg_sm = smf.logit('Direction ~ Lag1 + Lag2 + Lag3 + Lag4 + Lag5 + Volume',
					  data=data).fit()
logreg_sm.summary()
