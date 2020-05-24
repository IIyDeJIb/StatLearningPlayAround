import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt

# Generate data
x = np.arange(100)
y = 1 + x*0.03 + rnd.normal(0,1,len(x))

# Linear regression
X = np.stack((np.ones(len(x)), x)).transpose()
X = np.matrix(X)
b = np.linalg.inv(X.T * X)*X.T*y

# Hat matrix
H = X * np.linalg.inv(X.T * X)*X.T

# Plot leverages
plt.plot(H.diagonal().T)