

def abline(slope,intercept, ax=None):

	"""
	# Plot a line on the current axes using the intercept and slope
	:param slope:
	:param intercept:
	:return: matplotlib.lines.Line2D object
	"""
	import numpy as np
	import matplotlib.pyplot as plt

	if ax == None:
		currentAxes = plt.gca().axes
	else:
		currentAxes = ax

	x = np.array(currentAxes.get_xlim())
	y = slope*x + intercept

	return currentAxes.plot(x, y)
