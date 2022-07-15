from utils import *

def pltit(i, sgnl): # , axs, fotis_hits, acc):
	for j in [0,1]:
		axs[i][j].grid()
		axs[i][j].set_facecolor('lightgrey')
		axs[i][j].plot(np.arange(start = strt_wide, stop = stp_wide), sgnl[j][strt_wide:stp_wide], lw = .5)
		axs[i][j].fill_between(x = strt_stp, y1 = -3000*j - 15, y2 = 3000*j + 15, color = 'white')
		# plt.axhline(y = 13.100000000000001, xmin = sf[0], xmax = sf[1], color = 'red', lw = 1)
		if j:
			axs[i][j].set_ylabel("deg/sec")
			axs[i][j].set_title(name + " Ang " + "spike")
		else:
			axs[i][j].set_ylabel("g")
			axs[i][j].set_title(name + " Acc " + "spike")
		axs[i][j].legend(["X","Y","Z"])

def data_load(name, tag):
	subj = recording(name = name, tag = tag)
	subj.read_data()
	subj.filtering()
	subj.windowing(subj.filt_acc, subj.filt_ang, subj.spikes, wdlen)

	return subj.spikes, subj.filt_acc, subj.filt_ang, subj.windows
	# return subj.spikes, subj.raw_acc, subj.raw_ang, subj.windows


names = ["sltn", "gali", "sdrf", "pasx", "anti", "komi", "fot", "agge", "conp", "LH_galios"]
wdlen = window_length(names, sls)

names = ['sdrf', 'LH_galios']

fig, axs = plt.subplots(np.shape(names)[0], 2)
n = 0
for name in names:
	tag = '0' + str(n+1)
	spikes, acc, ang, windows = data_load(name, tag)
	sgnl = [acc, ang]
	a = 11 #np.random.randint(len(spikes))
	strt_wide = int(spikes[a, 0] - 400)
	stp_wide = int(spikes[a, 1] + 400)
	strt_stp = [int(spikes[a, 0]), int(spikes[a, 1])]
	pltit(n, sgnl)
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