import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

def labeling(dat, k, sbsr):
	#axs[k].step(np.arange(start = 1000, stop = 1400), dat[1000:1400], lw=.5)
	axs[k].plot(np.arange(len(dat)), dat, lw=.5)
	axs[k].set_facecolor('lightgrey')
	j = 1
	if sbsr == 0:
		axs[k].set_title("Spikes")
		ann_file = spks # [0:1]
	elif sbsr == 1:
		axs[k].set_title("Blocks")
		ann_file = blcks # [0:1]
	else:
		axs[k].set_title("Services")
		ann_file = srvs 
	
	for i in ann_file:
		axs[k].fill_between(x = i, y1 = -3000*k - 15, y2 = 3000*k + 15, color = 'white')
		axs[k].annotate(str(j), (i[0] + (i[1]-i[0])/2 - 1,  -3000*k -15), color = 'black')
		j = j + 1

	axs[k].grid()
	axs[k].legend(["X","Y","Z"])
	# axs[0].set_ylabel("g")
	# axs[0].set_title("Accelerators 3D")

names = ["gali", "sdrf", "sltn", "pasx", "anti", "komi", "fot"]
name = names[5]
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
