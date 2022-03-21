# Functions for filtering

from math import ceil, floor
# from statistics import mode, stdev
import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
from scipy.signal import lfilter, freqz, filtfilt, butter
from tkinter.font import BOLD

## Functions:
def read_data(name):
	datap = "C:\\Users\\30698\\Thesis_Fotis\\thesis\\data\\pickled_data\\" + name + ".pkl"
	
	with open(datap, 'rb') as f:
		data = pkl.load(f)

	spikes = data[0]
	blocks = data[1]
	services = data[2]

	datar = "C:\\Users\\30698\\Thesis_Fotis\\thesis\\data\\recordings\\" + name
	data1 = np.fromfile(datar+"acc.bin", dtype=np.int16, sep='')
	data2 = np.fromfile(datar+"angularrate.bin", dtype=np.int16, sep='')

	acc = data1.reshape(len(data1)//3, 3)
	ang = data2.reshape(len(data2)//3, 3)

	acc = acc * 0.0004
	ang = ang * 0.07

	spks = spikes[:, [1,2]]
	blcks = blocks
	srvs = services

	t = np.arange(len(acc)) / 64
	return acc, ang, t, spks, blcks, srvs

def mav_filter(signal, c):
	ret = signal.copy()
	w = ceil(64 * c)
	for i in np.arange(3):
		ret[:, i] = np.convolve(signal[:, i], np.ones(w)/w, 'same')
	
	# for i in np.arange(3):
	# 	temp = np.convolve(signal[:, i], np.ones(w)/w, 'valid')
	# 	ret[:, i] = np.concatenate( (np.zeros(len(signal[:, i]) - len(temp)), temp), axis = None)

	return ret

def hp_filter(signal, fs):
	# 1st method - Butter parameters:
	nyq = fs / 2
	cutoff = 1 / nyq
	b, a = butter(6, cutoff, btype='high', analog=False)
	y1 = signal.copy()
	# y2 = signal.copy()
	for i in np.arange(3):
		y1[:, i] = filtfilt(b, a, signal[:, i])
		# y2[:, i] = lfilter(b, a, acc_smth[:, i])

	# # 2nd method - Laplace filter:
	# h = np.array([1, -2, 1])
	# y1 = lfilter(h, [1.0], acc_smth)
	# y1 = np.convolve(acc_smth[:, 1], h, 'same')
	
	return y1

def extract_spikes_signal(signal, spikes):
	spks_signal = signal.copy()
	spks_spaces = signal.copy()
	k = 0
	j = 0
	samples = 64
	spaces = 150
	for s in spikes:
		i = floor(s[0])*samples
		bound = ceil(s[1])*samples
		# print(i, bound)
		while i <= bound:
			spks_signal[k, :] = signal[i, :]
			spks_spaces[j, :] = signal[i, :]
			k += 1
			j += 1
			i += 1
		spks_spaces[j:j+spaces, :] = np.zeros((spaces, 3))
		j = j+spaces

	spks_signal = spks_signal[0:k, :]
	spks_spaces = spks_spaces[0:j, :]
	return spks_signal, spks_spaces

def plot_fandraw(raw, filtered, t):
	fig, axs = plt.subplots(3)
	fig.suptitle('Raw and filtered signal')
	for i in np.arange(3):
		axs[i].plot(t, raw[:, i], color = 'red', linewidth = .75)
		axs[i].plot(t, filtered[:, i], color = 'blue', linewidth = .75, alpha = 0.5)
		axs[i].grid()

def plot_hist(signal, num_of_bins, n):
	if n == 1:
		fig, axs = plt.subplots(1)
		fig.suptitle('Time length of all spikes')
		axs.hist(signal, bins = num_of_bins, density = True, facecolor = '#040288', edgecolor='#EADDCA', linewidth=0.5)
		plt.xlabel('Time (sec)') 
		plt.ylabel('Multitude')
	elif n > 1:
		fig, axs = plt.subplots(n)
		fig.suptitle('Filtered Signal Histogram')
		for i in np.arange(n):
			axs[i].hist(signal[:, i], bins = num_of_bins, density = True, facecolor = '#040288', edgecolor='#EADDCA', linewidth=0.5)

def plot_spikes(spks_signal, spks_spaces):
	fig, axs = plt.subplots(3, 2)
	for i in np.arange(3):
		axs[i][0].plot(spks_signal[:, i], linewidth = 0.5)
		axs[i][1].plot(spks_spaces[:, i], color = 'magenta', linewidth = 0.5)

	axs[0][0].set_title('Continuous spikes')
	axs[0][1].set_title('Spaces between spikes')

def filtering(name):
	acc, ang, t, spk, blk, srv = read_data(name)
	# (2) Calculations:
	spk_dif = spk[:,1] - spk[:,0]
	# spk_mean = np.mean(spk_dif)
	# spk_max = np.max(spk_dif)
	# print(spk_max)
	# plt.scatter(np.arange(len(spk_dif)), spk_dif)
	# plt.show()


	## Signal filtering:
	# (1) Moving average filter:
	acc_smth = mav_filter(acc, c = 0.25)
	ang_smth = mav_filter(ang, c = 0.25)
	# (2) High-pass filter:
	facc = hp_filter(acc_smth, fs = 64)
	fang = hp_filter(ang_smth, fs = 64)


	## Construct Spikes' Signal:
	acc_spk_cont, acc_spk_wide = extract_spikes_signal(facc, spk)
	ang_spk_cont, ang_spk_wide = extract_spikes_signal(fang, spk)

	return spk_dif, facc, acc_spk_cont, acc_spk_wide, fang, ang_spk_cont, ang_spk_wide

def spike_signals(mode):
	names = np.array(["gali", "sdrf", "sltn", "pasx", "anti"])
	spk_len = list()

	acc_spk_c = list()
	acc_spk_w = list()
	ang_spk_c = list()
	ang_spk_w = list()

	#return order: spk_dif, facc, acc_spk_cont, acc_spk_wide, fang, ang_spk_cont, ang_spk_wide
	for name in names:
		dif, _, a, b, _, c, d = filtering(name)
		spk_len.extend(dif)
		acc_spk_c.extend(a)
		acc_spk_w.extend(b)
		ang_spk_c.extend(c)
		ang_spk_w.extend(d)

	if mode == 'spk':
		return spk_len
	elif mode == 'acc':
		return acc_spk_c, acc_spk_w
	elif mode == 'ang':
		return ang_spk_c, ang_spk_w
	elif mode == 'all':
		return spk_len, acc_spk_c, acc_spk_w, ang_spk_c, ang_spk_w
	else:
		print("Error: Wrong mode")
