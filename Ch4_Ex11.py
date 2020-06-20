import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, \
	QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

data = pd.read_csv('datasets\\Auto.csv')
data = data.assign(weight_x_accel=data['weight'] * data['acceleration'])

data['mpg01'] = data['mpg'].apply(lambda x: 1 if x > np.median(data['mpg']) else 0)

fig, ax = plt.subplots(1, 3)
ax[0].scatter(data['weight'], data['mpg01'])
ax[1].scatter(data['displacement'], data['mpg01'])
ax[2].scatter(data['acceleration'], data['mpg01'])

ax[0].set_title('weight')
ax[1].set_title('displacement')
ax[2].set_title('acceleration')

X_train, X_test, y_train, y_test = train_test_split(
	data[['acceleration', 'weight', 'displacement', 'weight_x_accel']],
	data['mpg01'], test_size=0.25)

accuScore = pd.DataFrame(index=['logreg', 'lda', 'qda'], columns=['accuracyScore'])

# All methods perform well (accuracy score>0.9)
lda = LinearDiscriminantAnalysis().fit(X_train, y_train)
confusion_matrix(y_test, lda.predict(X_test))
accuScore.loc['lda'] = accuracy_score(y_test, lda.predict(X_test))

logreg = LogisticRegression().fit(X_train, y_train)
confusion_matrix(y_test, logreg.predict(X_test))
accuScore.loc['logreg'] = accuracy_score(y_test, logreg.predict(X_test))

qda = LinearDiscriminantAnalysis().fit(X_train, y_train)
confusion_matrix(y_test, qda.predict(X_test))
accuScore.loc['qda'] = accuracy_score(y_test, qda.predict(X_test))

accuScore.plot(kind='bar')

# Now let us see which k does best in knn model
from sklearn.neighbors import KNeighborsClassifier

accuScore_k = pd.Series(index=np.arange(1, 100), name='accuracy_score',
						dtype='float')
for ii in accuScore_k.index:
	accuScore_k[ii] = accuracy_score(y_test,
									 KNeighborsClassifier(n_neighbors=ii).fit(
										 X_train, y_train).predict(X_test))

# The curve flattens out when the number of neighbors approaches the the number of
# test points.
ax = accuScore_k.plot()
ax.set_xlabel('n_neighbors')
ax.set_ylabel('accuracy_score')
