import random as rand
from math import floor, ceil
import os
import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
from scipy.signal import lfilter, freqz, filtfilt, butter
from tkinter.font import BOLD

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

		self.windows = None
		self.labels = None
		self.timestamps = None

	def read_data(self):
		p = os.path.dirname(os.getcwd())
		p1 = p + '\\data\\pickled_data\\'
		p2 = p + '\\data\\recordings\\'
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
		labels = wds[:, 0, 0].copy()
		timestamps = labels.copy()
		acc_temp = np.vstack( ( acc, np.zeros([62, 3]) ) )
		ang_temp = np.vstack( ( ang, np.zeros([62, 3]) ) )
		i = 0
		j = 0
		s = 0

		while s <= s_tot:
			ls = s + wd_smpl_len
			wds[i, :, 0:3] = acc_temp[s:ls, :]
			wds[i, :, 3:] = ang_temp[s:ls, :]
			timestamps[i] = s / 64
			step, labels[i] = self.find_step_and_label(spks, j, wd_smpl_len, s)
			
			i += 1
			s = s + step

		## Finding the unused slots in the rear of wds
		zer = np.zeros([65, 3])
		c = 1
		while wds[-c, :, :].any() == zer.any():
			c += 1

		## We exclude the last window, losing a constructed non-spike
		## window (mixed signal-tuples with zero-tuples)
		wds = wds[1:-c-1, :, :]
		labels = labels[1:-c-1]
		self.windows = wds
		self.labels = labels
		self.timestamps = timestamps
