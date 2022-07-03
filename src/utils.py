import random as rand
from math import floor, ceil
import os
import numpy as np
import pandas as pd
import pickle as pkl
from scipy.signal import lfilter, freqz, filtfilt, butter
from tkinter.font import BOLD
from filterfuncs import window_length
from copy import deepcopy
import sys
# print(len(sys.argv))
# if int(sys.argv[1]) == 1:
if len(sys.argv) > 1:
	import matplotlib
	matplotlib.use('TkAgg')
	from matplotlib import pyplot as plt
	sls = '/'
	print('We are in Ubuntu!')
else:
	import matplotlib.pyplot as plt
	sls = '\\'

# sls = sys.argv[1]

class recording:

	def __init__(self, name, tag):
		self.name = name
		self.tag = tag
		self.fs = 64
		self.raw_acc = None
		self.raw_ang = None
		self.t = None
		self.spikes = None
		self.filt_acc = None
		self.filt_ang = None
		if name.find('LH') == -1:
			self.left_hand = False  #default
		else:
			self.left_hand = True

		self.windows = None
		self.labels = None
		self.timestamps = None

	def set_signals(self, racc, rang, t, spks, facc, fang):
		self.raw_acc = racc
		self.raw_ang = rang
		self.t = t
		self.spikes = spks
		self.filt_acc = facc
		self.filt_ang = fang

	def set_windows(self, wds, lbls, tmst):
		self.windows = wds
		self.labels = lbls
		self.timestamps = tmst

	def copy(self):
		q = recording(self.name, self.tag)
		q.set_signals(self.raw_acc, self.raw_ang, self.t, self.spikes, self.filt_acc, self.filt_ang)
		q.set_windows(self.windows, self.labels, self.timestamps)
		return q

	def read_data(self):
		p = os.path.dirname(os.getcwd())
		p1 = p + sls + 'data' + sls + 'pickled_data' + sls
		p2 = p + sls + 'data' + sls + 'recordings' + sls
		datap = p1 + self.name + ".pkl"
		
		with open(datap, 'rb') as f:
			data = pkl.load(f)

		spikes = data[0]
		# blocks = data[1]
		# services = data[2]

		datar = p2 + self.name
		data1 = np.fromfile(datar+"acc.bin", dtype=np.int16, sep='')
		data2 = np.fromfile(datar+"angularrate.bin", dtype=np.int16, sep='')

		acc = data1.reshape(len(data1)//3, 3)
		ang = data2.reshape(len(data2)//3, 3)

		acc = acc * 0.0004
		ang = ang * 0.07

		spks = spikes[:, [1,2]]
		# blcks = blocks
		# srvs = services
		t = np.arange(len(acc)) / 64

		spks = np.array(spks*self.fs)
		spks[:, 0] = np.round(spks[:, 0])
		spks[:, 1] = np.ceil(spks[:, 1])
		
		# Hand mirroring for left-handed subjects. Converting their 
		# spike signals to an identical right-handed subject's signal
		if self.left_hand:
			acc, ang = hand_mirroring_signals(acc, ang)

		self.t = t
		self.raw_acc = acc
		self.raw_ang = ang
		self.spikes = spks.astype(int)

	def mav_filter(self, signal, c):
		ret = signal.copy()
		w = int(np.ceil(64 * c))
		for i in np.arange(3):
			ret[:, i] = np.convolve(signal[:, i], np.ones(w)/w, 'same')
		
		return ret
	
	def hp_filter(self, signal, fs):
		# 1st method - Butter parameters:
		nyq = fs / 2
		cutoff = 1 / nyq
		b, a = butter(6, cutoff, btype='high', analog=False)
		y1 = signal.copy()

		for i in np.arange(3):
			y1[:, i] = filtfilt(b, a, signal[:, i])
		
		return y1

	def filtering(self):
		# (1) Moving average filter:
		acc_smth = self.mav_filter(self.raw_acc, c=.25)
		ang_smth = self.mav_filter(self.raw_ang, c=.25)
		# (2) High-pass filter:
		facc = self.hp_filter(acc_smth, self.fs)
		fang = self.hp_filter(ang_smth, self.fs)

		self.filt_acc = facc
		self.filt_ang = fang

	def find_step_and_label(self, spks_sample, j, wd_smpl_len, s):
		ls = s + wd_smpl_len
		flag = True
		e = round(0.2 * 64)
		ovl = 0.6
		while flag:
			bound = ls - round(spks_sample[j, 1])
			if bound < 0:
				if -bound <= wd_smpl_len: # ls > floor(spks_sample[j, 0]):
					step = -bound
				else:
					step = round(wd_smpl_len*ovl)
					# step = floor(wd_smpl_len*0.4)
				flag = False
				label = 0
			elif bound <= e: # bound >= 0 and bound <= e:
				step = 1
				flag = False
				label = 1
			else:
				if j < len(spks_sample[:, 0])-1:
					j+=1
					# print(j)
				else:
					step = round(wd_smpl_len*ovl)
					# step = floor(wd_smpl_len*0.4)
					flag = False
					label = 0

		return step, label

	def windowing(self, acc, ang, spks, wd_smpl_len):
		s_tot = ceil(len(acc[:, 0]))
		## The maximum length of windows = total samples
		## Each window will contain wd_smpl_len samples
		wds = np.zeros([s_tot, wd_smpl_len, 6])
		labels = deepcopy(wds[:, 0, 0])
		timestamps = deepcopy(labels)
		acc_temp = np.vstack( ( acc, np.zeros([62, 3]) ) )
		ang_temp = np.vstack( ( ang, np.zeros([62, 3]) ) )
		i = 0
		j = 0
		s = 0

		while s <= s_tot:
			ls = s + wd_smpl_len
			wds[i, :, 0:3] = deepcopy(acc_temp[s:ls, :])
			wds[i, :, 3:] = deepcopy(ang_temp[s:ls, :])
			timestamps[i] = s / 64
			step, labels[i] = self.find_step_and_label(spks, j, wd_smpl_len, s)
			
			i += 1
			s = s + step

		## Finding the unused slots in the rear of wds
		zer = np.zeros([wd_smpl_len, 3])
		c = 1
		while wds[-c, :, :].any() == zer.any():
			c += 1

		## We exclude the last window, losing a constructed non-spike
		## window (mixed signal-tuples with zero-tuples)
		wds = wds[1:-c, :, :]
		labels = labels[1:-c]
		timestamps = timestamps[1:-c]
		self.windows = deepcopy(wds)

		# fig, axs = plt.subplots(2)
		# axs[0].plot(acc[-44:, :])
		# axs[1].plot(self.windows[-1, :, 0:3])
		# plt.show()

		self.labels = deepcopy(labels)
		self.timestamps = deepcopy(timestamps)

	def extend_windows(self, extra, n):
		self.windows = np.append(self.windows, extra, axis=0)
		self.labels  = np.append(self.labels, np.ones(n, dtype=np.int16))
		self.timestamps = np.append(self.timestamps, np.zeros(n, dtype=np.float64))




# collect all subjects in an object:
def subj_tot(names):
	print(f'\n# Before augmentation:')
	wd_smpl_len = window_length(names, sls)
	# print(wd_smpl_len)
	subjects = list()
	ns = list()
	po = list()
	ne = list()
	pos = 0
	neg = 0
	for i in np.arange(len(names)):
		n = names[i]
		if i < 9:
			t = '0' + str(i+1)
		else:
			t = str(i+1)
		subjects.append(recording(name = n, tag = t))

		subjects[i].read_data()
		subjects[i].filtering()

		# print(f'{np.shape(subjects[i].filt_acc)}')
		subjects[i].windowing(subjects[i].filt_acc, subjects[i].filt_ang, subjects[i].spikes, wd_smpl_len)

		unique, counts = np.unique(subjects[i].labels, return_counts=True)
		d = dict(zip(unique, counts))
		print(f'\n  Subject_{subjects[i].tag}:')
		print(f'    Negative windows: {d[0.0]}')
		print(f'    Positive windows: {d[1.0]}')
		neg += d[0.0]
		pos += d[1.0]
		ne.append(d[0.0])
		po.append(d[1.0])
		ns.append(d[0.0]//d[1.0])
		# ns.append(d[0.0]/d[1.0])

	print(f'\n  Total:\n    Positives = {pos}\n    Negatives = {neg}')
	return subjects, ns, po, ne


# input: all windows -> (100.000 windows, 48 samples, 6 sensors)
# output: mean and std for each sensor
def standardization_parameters(windows):

# Equal Method for extracting means and std:
#	Instead of separated-windows array of (100.000, 48, 6),
#	we create the merged-windows array of (100.000*48, 6):
	
	# d1, d2, d3 = np.shape(windows)
	# nwds = np.empty([d1*d2, d3], dtype = np.float64)
	# for i in np.arange(windows.shape[0]):
	# 	nwds[ i*d2 : (i+1)*d2, : ] = windows[i, :, :]
	
	# means = np.mean(nwds, axis = 0)
	# stds = np.std(nwds, axis = 0)

# Without merging windows: (Equal method for means, stds)
	means = np.mean(windows, axis = (0,1))
	stds = np.std(windows, axis = (0,1))
	
	return means, stds


def apply_stadardization(windows, means, stds):
	# a = deepcopy(windows)
	for i in np.arange(windows.shape[2]):
		windows[:, :, i] = (windows[:, :, i] - means[i]) / stds[i]

	# print(f'{np.mean(windows, axis = (0,1))}')
	# print(np.std(windows, axis = (0,1)))
	# print(f'Original means: {means}')
	# print(f'Original stds: {stds}')
	return windows


## Positive class augmentation:
def rand_rot(theta):
	Ry = np.array([
		[np.cos(theta), 0, np.sin(theta)],
		[0, 1, 0],
		[-np.sin(theta), 0, np.cos(theta)]
	])
	Rz = np.array([
		[np.cos(theta), -np.sin(theta), 0],
		[np.sin(theta), np.cos(theta), 0],
		[0, 0, 1]
	])

	r = rand.randint(1, 4)
	if r == 1:
		R = Ry
	elif r == 2:
		R = Rz
	elif r == 3:
		R = np.dot(Ry,Rz)
	elif r == 4:
		R = np.dot(Rz,Ry)

	# print(f'{np.dot(Ry,Rz)}\n\n{np.dot(Rz,Ry)}\n')

	zer = np.zeros( (3, 3) )
	Q1 = np.vstack( (R, zer) )
	Q2 = np.vstack( (zer, R) )
	Q = np.hstack( (Q1, Q2) )
	# print(Q)

	return Q


## Spikes-augmentation: Balance negative and positive class
def balance_windows(subjects, ns, posi, neg):
	newsubj = deepcopy(subjects)
	s_ind = 0
	for s, n, pos, neg in zip(subjects, ns, posi, neg):
		lst = np.round(np.random.normal(loc=0.0, scale=10, size = n+1), 3)
		ind = np.transpose( np.nonzero(s.labels) )
		_, d2, d3 = np.shape(s.windows)
		temp = np.empty( [(n+1)*pos, d2, d3] )
		j = 0
		for theta in lst:
			# theta = np.pi/2
			Q = rand_rot(theta)
			for i in ind:
				## (i)  Counter-Clockwise rotation:
				temp[j] = np.dot(np.squeeze(s.windows[i, :, :]), Q)

				## (ii) Clockwise rotation:
				# temp[j] = np.transpose( np.dot(Q, np.squeeze(np.transpose(s.windows[i, :, :])) ) )
				
				j += 1

		aug_end = neg - pos
		temp = temp[:aug_end, :, :]
		# print(f'Before: {np.shape(newsubj[s_ind].windows)}')
		# newsubj[s_ind].extend_windows(temp, n*pos)
		newsubj[s_ind].extend_windows(temp, aug_end)
		s_ind += 1
		# print(f'After: {np.shape(newsubj[s_ind-1].windows)}')

	print(f'\n# After augmentation:')
	neg = 0
	pos = 0
	for nsj in newsubj:
		unique, counts = np.unique(nsj.labels, return_counts=True)
		d = dict(zip(unique, counts))
		print(f'\n  Subject_{nsj.tag}:')
		print(f'    Negative windows: {d[0.0]}')
		print(f'    Positive windows: {d[1.0]}')
		neg += d[0.0]
		pos += d[1.0]
	
	print(f'\n  Total:\n    Positives = {pos}\n    Negatives = {neg}')
	return newsubj, ind


## Hand mirroring methods:
def hand_mirroring_signals(acc, ang):
	B = np.array([
		[-1, 0, 0],
		[0, 1, 0],
		[0, 0, 1]
	])

	acc = acc @ B
	ang = ang @ B

	return acc, ang
	
def hand_mirroring_windows(wds):
	B = np.array([
		[-1, 0, 0, 0, 0, 0],
		[0, 1, 0, 0, 0, 0],
		[0, 0, 1, 0, 0, 0],
		[0, 0, 0, -1, 0, 0],
		[0, 0, 0, 0, 1, 0],
		[0, 0, 0, 0, 0, 1]
	])

	b = wds @ B
	
	if np.array_equal(b.shape, wds.shape):
		return b
	else:
		print("Error on shape!")



#################################### FILTERFUNCS ####################################