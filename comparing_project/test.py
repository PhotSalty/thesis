import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from matplotlib import pyplot as plt
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, accuracy_score
import pickle as pkl
from utils_case import extract_indicators

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

	df = pd.DataFrame(target, columns = ['target'])
	df['predicted'] = pred
	print(df)

	cm = confusion_matrix(target, pred, labels = [0, 1])
	cm = confusion_matrix(target, pred)
	print(cm)
	
	tn, fp, fn, tp = cm.ravel()

	wgt = 97/3
	is1 = np.count_nonzero(target)
	print(is1)
	wgt1 = (np.shape(target)[0] - is1) / is1

	print(f'\nWeight = {wgt}, calculated = {wgt1}\n')
	# wacc_cst = (cm[0, 0]*wgt + cm[1, 1]) / ( (cm[0, 0] + cm[1, 0])*wgt + cm[1, 1] + cm[0, 1] )
	wacc_cst = (tp*wgt + tn) / ( (tp + fn)*wgt + tn + fp )

	wacc2 = ( (tp / (tp + fn)) + (tn / (tn + fp)) ) / 2

	acc = accuracy_score(target, pred)
	wacc = balanced_accuracy_score(target, pred)
	print(acc, wacc, wacc_cst, wacc2)


def indicator_indices():

	pkl_path = 'preprocessed_data\\prepdata.pkl'
	with open(pkl_path, 'rb') as f:
		s_orig = pkl.load(f)
		s_auxi = pkl.load(f)
		labels = pkl.load(f)
		e_impact = pkl.load(f)
		swing_interval = pkl.load(f)

	a, b, c = extract_indicators(s_orig[0, 0], s_auxi[0, 0], labels[0], e_impact)

	print(f'{c}')
	print(s_auxi[0, 0][c])
	print(c.shape[0])
	print(e_impact)
	flag = True
	for s in s_auxi[0, 0][c]:
		if s < e_impact:
			flag = False
	
	print(flag)


def smote_ovrsmpl():

	from copy import deepcopy
	from collections import Counter
	from imblearn.over_sampling import SMOTE

	x = np.random.randint(0, 5, size = (100, 6, 3))
	target = np.zeros(100, dtype = np.int8)
	target[1], target[3], target[19], target[23], target[30], target[44], target[67], target[82] = 1, 1, 1, 1, 1, 1, 1, 1

	print(Counter(target))
	oversample = SMOTE()
	x1, y1 = oversample.fit_resample(x[:,:,0], target)
	x2, y2 = oversample.fit_resample(x[:,:,1], target)
	x3, y3 = oversample.fit_resample(x[:,:,2], target)
	
	x_aug = np.zeros((x1.shape[0], x1.shape[1], 3))
	x_aug[:, :, 0] = x1
	x_aug[:, :, 1] = x2
	x_aug[:, :, 2] = x3
	
	if np.array_equal(y1, y2) and np.array_equal(y1, y3):
		y_aug = y1
	else:
		print('\nError on y augmentation\n')

	print(x_aug.shape, y_aug.shape)
	print(x_aug[100:110, :, :])
	print(y_aug[100:110])

	print(Counter(y1))

def randnorminit():
	from keras.initializers import RandomNormal
	initializer = RandomNormal(mean = 0.0, stddev = 0.1)
	val = initializer(shape = (2,2))
	print(val)


def array_out_of_arrays():

	a = np.zeros(5, dtype = object)
	b = np.ones((10))
	c = np.zeros((0))
	# b = np.ones((10, 3))
	# c = np.zeros((0, 3))
	print(b.shape, c.shape)
	for i in np.arange(5):
		a[i] = b * i
	
	for i in np.arange(5):
		# c = np.append(c, a[i], axis = 0)
		c = np.append(c, a[i])

	print(c)
	print(c[10:20])

# array_out_of_arrays()


def counter_weights():

	from collections import Counter

	a = np.zeros(20)
	ind = np.random.randint(0, 20, size = 5)
	a[ind] = 1
	d = {
		0. : 15.,
		1. : 5.
	}
	print(Counter(a))
	print(dict(Counter(a)))
	print(d)

counter_weights()