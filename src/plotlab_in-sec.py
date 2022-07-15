# import numpy as np
# import pickle as pkl
# import matplotlib.pyplot as plt
# import os
# from utils import set_ubuntu_slash
# import sys

# if int(sys.argv[1]) == 1:
# 	import matplotlib
# 	matplotlib.use('TkAgg')
# 	from matplotlib import pyplot as plt
# 	sls = '/'
# 	print('We are in Ubuntu!')
# else:
# 	import matplotlib.pyplot as plt
# 	sls = '\\'

from utils import *

# sls = sys.argv[1]

def labeling(dat, k, sbsr):
	#axs[k].step(np.arange(start = 1000, stop = 1400), dat[1000:1400], lw=.5)
	# axs[k].plot(np.arange(len(dat)), dat, lw=.5)
	freq = 64
	t = np.arange(len(dat)/freq, step = 1/freq)
	axs[k].plot(t, dat, lw=.5)
	axs[k].set_facecolor('lightgrey')
	j = 1
	if sbsr == 0:
		axs[k].set_title("Spikes")
		ann_file = spks/freq # [0:1]
	elif sbsr == 1:
		axs[k].set_title("Blocks")
		ann_file = blcks/freq # [0:1]
	else:
		axs[k].set_title("Services")
		ann_file = srvs/freq 
	
	for i in ann_file:
		axs[k].fill_between(x = i, y1 = -3000*k - 15, y2 = 3000*k + 15, color = 'white')
		axs[k].annotate(str(j), (i[0] + (i[1]-i[0])/2 - 1,  -3000*k -15), color = 'black')
		j = j + 1

	axs[k].grid()
	axs[k].legend(["X","Y","Z"])
	# axs[0].set_ylabel("g")
	# axs[0].set_title("Accelerators 3D")

names = ["sltn", "gali", "sdrf", "pasx", "anti", "komi", "fot", "agge", "conp", "LH_galios"]
name = names[7]

p = os.path.dirname(os.getcwd())

p1 = p + sls + 'data' + sls + 'pickled_data' + sls
datap = p1 + name + ".pkl"
# datap = p1 + name + '_old' + ".pkl"

with open(datap, 'rb') as f:
	data = pkl.load(f)

spikes = data[0]
blocks = data[1]
services = data[2]

p2 = p + sls + 'data' + sls + 'recordings' + sls

datar = p2 + name
data1 = np.fromfile(datar+"acc.bin", dtype=np.int16, sep='')
data2 = np.fromfile(datar+"angularrate.bin", dtype=np.int16, sep='')

acc = data1.reshape(len(data1)//3, 3)
ang = data2.reshape(len(data2)//3, 3)

acc = acc * 0.0004
ang = ang * 0.07

if name.find('LH') != -1:
	print("left handed subject detected")
	acc, ang = hand_mirroring_signals(acc, ang)

# Spikes has 4 columns:
# arm-swing, last ground contact, first ground contact, 2 feet on the ground
# [0, 1, 2, 3] START: [0 or 1] , END: [2 or 3]
spks = spikes[:, [1,2]] * 64
blcks = blocks * 64
srvs = services * 64

s_b_sr = [0, 1, 2]
for i in s_b_sr:
	fig, axs = plt.subplots(2)
	labeling(acc, 0, i)
	labeling(ang, 1, i)

plt.show()
