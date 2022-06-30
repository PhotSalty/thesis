import numpy as np
import pickle as pkl
import os
from utils import set_ubuntu_slash
import sys
if int(sys.argv[1]) == 1:
	import matplotlib
	matplotlib.use('TkAgg')
	from matplotlib import pyplot as plt
	sls = '/'
	print('We are in Ubuntu!')
else:
	import matplotlib.pyplot as plt
	sls = '\\'

# sls = sys.argv[1]

def pltit(i): # , axs, fotis_hits, acc):
	for j in [0,1]:
		axs[i][j].grid()
		axs[i][j].set_facecolor('lightgrey')
		axs[i][j].plot(np.arange(start = sf[0], stop = sf[1]), dat[j][sf[0]:sf[1]], lw = .5)
		axs[i][j].fill_between(x = sf1, y1 = -3000*j - 15, y2 = 3000*j + 15, color = 'white')
		# plt.axhline(y = 13.100000000000001, xmin = sf[0], xmax = sf[1], color = 'red', lw = 1)
		if j:
			axs[i][j].set_ylabel("deg/sec")
			axs[i][j].set_title(name + " Ang " + "spike")
		else:
			axs[i][j].set_ylabel("g")
			axs[i][j].set_title(name + " Acc " + "spike")
		axs[i][j].legend(["X","Y","Z"])

def data_load(name):
	p = os.path.dirname(os.getcwd())

	# sls = set_ubuntu_slash(sys.argv[0])
	
	n = p + sls + 'data' + sls
	# n = "C:\\Users\\30698\\Thesis_Fotis\\thesis\\data\\"
	datap = n + 'pickled_data' + sls + name + ".pkl"
	flacc = n + "recordings" + sls + name + "acc.bin"
	flang = n + "recordings" + sls + name + "angularrate.bin"

	with open(datap, 'rb') as f:
		data = pkl.load(f)

	spikes = data[0]
	#blocks = data[1]
	#services = data[2]

	spks = spikes[:, [1,2]] * 64

	data1 = np.fromfile(flacc, dtype=np.int16, sep='')
	data2 = np.fromfile(flang, dtype=np.int16, sep='')

	acc = data1.reshape(len(data1)//3, 3)
	ang = data2.reshape(len(data2)//3, 3)

	acc = acc * 0.0004
	ang = ang * 0.07

	return spks, acc, ang


names = ["sltn", "gali", "sdrf", "pasx", "anti", "komi", "fot", "agge", "conp"]
names = ["sdrf", "agge", "conp"]

fig, axs = plt.subplots(np.shape(names)[0], 2)
n = 0
for name in names:
	spikes, acc, ang = data_load(name)
	dat = [acc, ang]
	a = 15 #np.random.randint(len(spikes))
	sf = [int(spikes[a, 0] - 400), int(spikes[a, 1] + 400)]
	sf1 = [int(spikes[a, 0]), int(spikes[a, 1])]
	pltit(n)
	n += 1


plt.show()


# print(np.where(acc[1] == np.amax(acc[1])))
#print(np.amax(acc[:,0]), np.amax(acc[:,1]), np.amax(acc[:,2]))

# # Spikes has 4 columns:
# # arm-swing, last ground contact, first ground contact, 2 feet on the ground
# # [0, 1, 2, 3] START: [0 or 1] , END: [2 or 3]
# spks = spikes[:, [1,2]] * 64
# blcks = blocks * 64
# srvs = services * 64

# s_b_sr = [0, 1, 2]
# for i in s_b_sr:
# 	fig, axs = plt.subplots(2)
# 	cmp(acc, 0, i)
# 	cmp(ang, 1, i)

# plt.show()