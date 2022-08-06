import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

def dataframe_test():
	names = np.array(['sltn', 'gali', 'sdrf', 'pasx', 'anti', 'komi', 'fot', 'agge', 'conp', 'LH_galios'])
	a = np.zeros((2, 10))
	a = np.random.random((2, 10))
	# df = pd.DataFrame(data = aa, columns = names, index = ['start', 'stop'])
	df = pd.DataFrame(data = a, columns = names, index = ['Signal_start', 'Signal_stop'])
	print(df)

	# df.iloc[0, 2] = 5
	df = df.round(2)

def local_maxima():
	q = np.random.randint(low = 0, high = 10, size = (20))
	print(q)

	maxima = argrelextrema(q, np.greater)
	print(maxima)

	print(maxima[0])
	print(q[maxima[0][0]])


def mean2d():
	q = np.random.randint(low = 0, high = 10, size = (20, 3))
	df = pd.DataFrame(q, columns = ['x', 'y', 'z'])
	print(df)

	ind = np.arange(start = 8, stop = 16)
	print(ind)

	print(q[ind])
	outp = np.mean(q[ind], axis = 0)
	x, y, z = outp
	print(x, y, z)
	print(outp)


def conc2d():
	a = np.zeros((1,3))
	b = np.ones((1,3))
	c = np.concatenate((a, b), axis = 1)
	print(c)

conc2d()