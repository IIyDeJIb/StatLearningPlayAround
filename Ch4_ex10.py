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
confMat = confusion_matrix(y, y_pred)

# Fit with statsmodels to see R-like summary
logreg_sm = smf.logit('Direction ~ Lag1 + Lag2 + Lag3 + Lag4 + Lag5 + Volume',
					  data=data).fit()
logreg_sm.summary()

# --- Use just Lag2, split train and tests
y_train = data.loc[data['Year'] < 2009, 'Direction'].map({'Up': 1, 'Down': 0})
y_test = data.loc[data['Year'] >= 2009, 'Direction'].map({'Up': 1, 'Down': 0})
X_train = pd.DataFrame(data.loc[data['Year'] < 2009, 'Lag2'])
X_test = pd.DataFrame(data.loc[data['Year'] >= 2009, 'Lag2'])

logreg_Lag2 = LogisticRegression().fit(X_train, y_train.values.ravel())

y_pred = logreg_Lag2.predict(X_test)
confusion_matrix(y_test, y_pred)

# Plot the logistic curve
import matplotlib.pyplot as plt
from numpy import linspace
from scipy.special._ufuncs import expit

fig, ax = plt.subplots()
ax.scatter(X.loc[800:, 'Lag2'], y.loc[800:])
ax.grid()
ax.set_xlim(left=-80, right=80)
ax.get_children()
ax.plot(linspace(-80, 80, 1000), expit(
	logreg_Lag2.coef_ * linspace(-80, 80, 1000) + logreg_Lag2.intercept_).flatten(),
		color='red')
ax.set_title('Logistic Regression', fontsize=14)
ax.set_xlabel('Lag2')
ax.set_ylabel('Direction Up/Down')