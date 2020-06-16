import numpy as np
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

data['Direction'] = data['Direction'].map({'Up': 1, 'Down': 0})
y = data['Direction']
logref_sk = LogisticRegression().fit(X, y)
y_pred = logref_sk.predict(X)
confMat = confusion_matrix(y, y_pred)

# Fit with statsmodels to see R-like summary
logreg_sm = smf.logit('Direction ~ Lag1 + Lag2 + Lag3 + Lag4 + Lag5 + Volume',
					  data=data).fit()
logreg_sm.summary()

# --- Use just Lag2, split train and tests
y_train = data.loc[data['Year'] < 2009, 'Direction']
y_test = data.loc[data['Year'] >= 2009, 'Direction']
X_train = pd.DataFrame(data.loc[data['Year'] < 2009, 'Lag2'])
X_test = pd.DataFrame(data.loc[data['Year'] >= 2009, 'Lag2'])

logreg_Lag2 = LogisticRegression().fit(X_train, y_train.values.ravel())

y_pred_Lag2 = logreg_Lag2.predict(X_test)
confusion_matrix(y_test, y_pred_Lag2)

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

# Fit LDA and QDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis().fit(X_train, y_train)
y_pred_lda = lda.predict(X_test)
confusion_matrix(y_test, y_pred_lda)

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

qda = QuadraticDiscriminantAnalysis().fit(X_train, y_train)
y_pred_qda = qda.predict(X_test)
confusion_matrix(y_test, y_pred_qda)

from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred_qda)
accuracy_score(y_test, y_pred_lda)
accuracy_score(y_test, y_pred_Lag2)

# Fit Knn
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1).fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
confusion_matrix(y_test, y_pred_knn)

# Apply transformations to the predictors and see how that affects model accuracy
# For simplicity, I will only track accuracy scores here and will apply
# transformations only to Volume.

accuScores = pd.DataFrame(
	index=['original', 'Lag1_x_Volume', 'Lag2_x_Volume',
		   'Lag1_x_Lag2_x_Volume',
		   'Lag1_x_Lag2'],
	columns=['original', 'log', 'sqr']
)

# List of transformations

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data.loc[:, 'Lag1':'Volume'],
													data['Direction'],
													test_size=0.25)
# no interations
accuScores.loc['original', 'original'] = accuracy_score(
	y_test, LogisticRegression().fit(
		X_train, y_train).predict(
		X_test))

accuScores.loc['original', 'log'] = accuracy_score(
	y_test, LogisticRegression().fit(
		X_train.apply(
			lambda x: np.log(x) if x.name == 'Volume' else x), y_train).predict(
		X_test.apply(
			lambda x: np.log(x) if x.name == 'Volume' else x)))

accuScores.loc['original', 'sqr'] = accuracy_score(
	y_test, LogisticRegression().fit(
		X_train.apply(
			lambda x: np.square(x) if x.name == 'Volume' else x), y_train).predict(
		X_test.apply(
			lambda x: np.square(x) if x.name == 'Volume' else x)))


# Lag1 * Volume
accuScores.loc['Lag1_x_Volume', 'original'] = accuracy_score(
	y_test, LogisticRegression().fit(
		X_train.assign(Lag1_x_Volume=X_train['Lag1'] * X_train['Volume']),
		y_train).predict(
		X_test.assign(Lag1_x_Volume=X_test['Lag1'] * X_test['Volume'])))

accuScores.loc['Lag1_x_Volume', 'log'] = accuracy_score(
	y_test, LogisticRegression().fit(
		X_train.assign(Lag1_x_Volume=X_train['Lag1'] * np.log(X_train['Volume'])),
		y_train).predict(
		X_test.assign(Lag1_x_Volume=X_test['Lag1'] * np.log(X_test['Volume']))))

accuScores.loc['Lag1_x_Volume', 'sqr'] = accuracy_score(
	y_test, LogisticRegression().fit(
		X_train.assign(Lag1_x_Volume=X_train['Lag1'] * np.square(X_train['Volume'])),
		y_train).predict(
		X_test.assign(Lag1_x_Volume=X_test['Lag1'] * np.square(X_test['Volume']))))

# Lag2 * Volume
accuScores.loc['Lag2_x_Volume', 'original'] = accuracy_score(
	y_test, LogisticRegression().fit(
		X_train.assign(Lag2_x_Volume=X_train['Lag2'] * X_train['Volume']),
		y_train).predict(
		X_test.assign(Lag2_x_Volume=X_test['Lag2'] * X_test['Volume'])))

accuScores.loc['Lag2_x_Volume', 'log'] = accuracy_score(
	y_test, LogisticRegression().fit(
		X_train.assign(Lag2_x_Volume=X_train['Lag2'] * np.log(X_train['Volume'])),
		y_train).predict(
		X_test.assign(Lag2_x_Volume=X_test['Lag2'] * np.log(X_test['Volume']))))

accuScores.loc['Lag2_x_Volume', 'sqr'] = accuracy_score(
	y_test, LogisticRegression().fit(
		X_train.assign(Lag2_x_Volume=X_train['Lag2'] * np.square(X_train['Volume'])),
		y_train).predict(
		X_test.assign(Lag2_x_Volume=X_test['Lag2'] * np.square(X_test['Volume']))))


# Lag1 * Lag2 * Volume
accuScores.loc['Lag1_x_Lag2_x_Volume', 'original'] = accuracy_score(
	y_test, LogisticRegression().fit(
		X_train.assign(
			Lag1_x_Lag2_x_Volume=X_train['Lag1'] * X_train['Lag2'] * X_train[
				'Volume']),
		y_train).predict(
		X_test.assign(Lag1_x_Lag2_x_Volume=X_test['Lag1'] * X_test['Lag2'] * X_test[
			'Volume'])))

accuScores.loc['Lag1_x_Lag2_x_Volume', 'log'] = accuracy_score(
	y_test, LogisticRegression().fit(
		X_train.assign(
			Lag1_x_Lag2_x_Volume=X_train['Lag1'] * X_train['Lag2'] * np.log(
				X_train['Volume'])),
		y_train).predict(
		X_test.assign(Lag1_x_Lag2_x_Volume=X_test['Lag1'] * X_test['Lag2'] * np.log(
			X_test['Volume']))))

accuScores.loc['Lag1_x_Lag2_x_Volume', 'sqr'] = accuracy_score(
	y_test, LogisticRegression().fit(
		X_train.assign(
			Lag1_x_Lag2_x_Volume=X_train['Lag1'] * X_train['Lag2'] * np.square(
				X_train['Volume'])),
		y_train).predict(
		X_test.assign(
			Lag1_x_Lag2_x_Volume=X_test['Lag1'] * X_test['Lag2'] * np.square(
				X_test['Volume']))))

# Lag1 * Lag2
accuScores.loc['Lag1_x_Lag2', 'original'] = accuracy_score(
	y_test, LogisticRegression().fit(
		X_train.assign(
			Lag1_x_Lag2=X_train['Lag1'] * X_train['Lag2']),
		y_train).predict(
		X_test.assign(Lag1_x_Lag2=X_test['Lag1'] * X_test['Lag2'])))

# accuScores.loc['Lag1_x_Lag2', 'log'] = accuracy_score(
# 	y_test, LogisticRegression().fit(
# 		X_train.assign(
# 			log_Lag1_x_Lag2=np.log(X_train['Lag1'] * X_train['Lag2'])),
# 		y_train).predict(
# 		X_test.assign(log_Lag1_x_Lag2=np.log(X_test['Lag1'] * X_test['Lag2']))))

accuScores.loc['Lag1_x_Lag2', 'sqr'] = accuracy_score(
	y_test, LogisticRegression().fit(
		X_train.assign(
			sq_Lag1_x_Lag2=np.square(X_train['Lag1'] * X_train['Lag2'])),
		y_train).predict(
		X_test.assign(
			sq_Lag1_x_Lag2=np.square(X_test['Lag1'] * X_test['Lag2']))))

# Altogether
accuScores.loc['Altogether', 'original'] = accuracy_score(
	y_test, LogisticRegression().fit(
		X_train.assign(
			Lag1_x_Volume=X_train['Lag1'] * X_train['Volume'],
			Lag2_x_Volume=X_train['Lag2'] * X_train['Volume'],
			Lag1_x_Lag2_x_Volume=X_train['Lag1'] * X_train['Lag2'] * X_train[
				'Volume'],
			Lag1_x_Lag2=X_train['Lag1'] * X_train['Lag2']),
		y_train).predict(
		X_test.assign(
			Lag1_x_Volume=X_test['Lag1'] * X_test['Volume'],
			Lag2_x_Volume=X_test['Lag2'] * X_test['Volume'],
			Lag1_x_Lag2_x_Volume=X_test['Lag1'] * X_test['Lag2'] * X_test[
				'Volume'],
			Lag1_x_Lag2=X_test['Lag1'] * X_test['Lag2'])))

accuScores.loc['Altogether', 'log'] = accuracy_score(
	y_test, LogisticRegression().fit(
		X_train.assign(Lag1_x_Volume=X_train['Lag1'] * np.log(X_train['Volume']),
			Lag2_x_Volume=X_train['Lag2'] * np.log(X_train['Volume']),
			Lag1_x_Lag2_x_Volume=X_train['Lag1'] * X_train['Lag2'] * np.log(X_train[
				'Volume']),
			Lag1_x_Lag2=X_train['Lag1'] * X_train['Lag2']),
		y_train).predict(
		X_test.assign(Lag1_x_Volume=X_test['Lag1'] * np.log(X_test['Volume']),
			Lag2_x_Volume=X_test['Lag2'] * np.log(X_test['Volume']),
			Lag1_x_Lag2_x_Volume=X_test['Lag1'] * X_test['Lag2'] * np.log(X_test[
				'Volume']),
			Lag1_x_Lag2=X_test['Lag1'] * X_test['Lag2'])))

# accuScores.loc['Altogether', 'sqr'] = accuracy_score(
# 	y_test, LogisticRegression().fit(
# 		X_train.assign(Lag1_x_Volume=X_train['Lag1'] * np.square(X_train['Volume']),
# 			Lag2_x_Volume=X_train['Lag2'] * np.square(X_train['Volume']),
# 			Lag1_x_Lag2_x_Volume=X_train['Lag1'] * X_train['Lag2'] * np.square(X_train[
# 				'Volume']),
# 			Lag1_x_Lag2=X_train['Lag1'] * X_train['Lag2']),
# 		y_train).predict(
# 		X_test.assign(Lag1_x_Volume=X_test['Lag1'] * np.square(X_test['Volume']),
# 			Lag2_x_Volume=X_test['Lag2'] * np.square(X_test['Volume']),
# 			Lag1_x_Lag2_x_Volume=X_test['Lag1'] * X_test['Lag2'] * np.square(X_test[
# 				'Volume']),
# 			Lag1_x_Lag2=X_test['Lag1'] * X_test['Lag2'])))
