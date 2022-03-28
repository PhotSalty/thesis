import random as rand
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

	def read_data(self):
		datap = "C:\\Users\\30698\\Thesis_Fotis\\thesis\\data\\pickled_data\\" + self.name + ".pkl"
		
		with open(datap, 'rb') as f:
			data = pkl.load(f)

		spikes = data[0]
		# blocks = data[1]
		# services = data[2]

		datar = "C:\\Users\\30698\\Thesis_Fotis\\thesis\\data\\recordings\\" + self.name
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

def plot_fandraw(raw, filtered, t):
	fig, axs = plt.subplots(3)
	fig.suptitle('Raw and filtered signal')
	for i in np.arange(3):
		axs[i].plot(t, raw[:, i], color = 'red', linewidth = .75)
		axs[i].plot(t, filtered[:, i], color = 'blue', linewidth = .75, alpha = 0.5)
		axs[i].grid()


subject01 = recording(name = 'sltn', tag = '01')
subject01.read_data()
subject01.filtering()

plot_fandraw(subject01.raw_acc, subject01.filt_acc, subject01.t)
plt.show()
