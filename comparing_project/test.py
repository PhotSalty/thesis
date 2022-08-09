import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from matplotlib import pyplot as plt
import pickle as pkl

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

def plotpklsave():

	fig = np.zeros(5, dtype = object)
	for i in np.arange(5):

		a = np.linspace(0, 2000, 15*(i+1))
		b = np.sin(a)

		fig[i] = plt.figure(f'figure <{i}>')
		plt.plot(a, b)
		plt.title(f'<{i}>')

	with open('figtest.pkl', 'wb') as f:
		pkl.dump(fig, f)

	with open('figtest.pkl', 'rb') as f1:
		fig1 = pkl.load(f1)

	for i in np.arange(5):
		plt.figure(fig1[i])
		plt.close(fig[i])
	
	plt.show()

def waccur():
	
	# target = np.random.randint(0, 2, 100, dtype = np.int8)
	target = np.zeros(100, dtype = np.int8)
	target[1], target[23], target[82] = 1, 1, 1
	pred   = np.random.randint(0, 2, 100, dtype = np.int8)

	from sklearn.metrics import balanced_accuracy_score, confusion_matrix, accuracy_score

	cm = confusion_matrix(target, pred, labels = [0, 1])
	wgt = 97/3
	is1 = np.count_nonzero(target)
	print(is1)
	wgt1 = (np.shape(target)[0] - is1) / is1

	print(f'\nWeight = {wgt}, calculated = {wgt1}\n')
	wacc_cst = (cm[0, 0] + cm[1, 1]*wgt) / ( (cm[1, 1] + cm[1, 0])*wgt + cm[0, 0] + cm[0, 1] )

	acc = accuracy_score(target, pred)
	wacc = balanced_accuracy_score(target, pred)

	print(acc, wacc, wacc_cst)

waccur()