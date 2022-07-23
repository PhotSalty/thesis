
from utils import *

def plotit(i, sgnl):
# j = 0  ->  Accelerometer signal
# j = 1  ->  Angularrate signal
	for j in [0,1]:

		axs[i][j].grid()
		axs[i][j].set_facecolor('lightgrey')

	# Plot the wide part of the signal:
		axs[i][j].plot(np.arange(start = strt_wide, stop = stp_wide), sgnl[j][strt_wide:stp_wide], lw = .5)
		
	# Annotate white only the background of the spike-signal
		axs[i][j].fill_between(x = strt_stp, y1 = -3000*j - 15, y2 = 3000*j + 15, color = 'white')
	
	# Set labels
		if j:
			axs[i][j].set_ylabel('deg/sec')
			axs[i][j].set_title(name + ' Ang ' + 'spike')
		else:
			axs[i][j].set_ylabel('g')
			axs[i][j].set_title(name + ' Acc ' + 'spike')
		
		axs[i][j].legend(['X','Y','Z'])


# def data_load(name, tag, wdlen):
def data_load(name, tag):

	subj = recording(name = name, tag = tag)
	subj.read_data()
	subj.filtering()
	# subj.windowing(subj.filt_acc, subj.filt_ang, subj.spikes, wdlen)

	# return subj.spikes, subj.raw_acc, subj.raw_ang, subj.windows
	return subj.spikes, subj.filt_acc, subj.filt_ang, subj.windows


names = np.array(['sltn', 'gali', 'sdrf', 'pasx', 'anti', 'komi', 'fot', 'agge', 'conp', 'LH_galios'])

# Window length of all subjects
#window_length = calc_window_length(names)

# Indices of names to compare
cp_ind = np.asarray( [2, 9] )

names_cmpr = names[ cp_ind ]

fig, axs = plt.subplots(len(names_cmpr), 2)

for i, name in zip(np.arange(len(names_cmpr)), names_cmpr):

	if cp_ind[i] < 9:
		tag = '0' + str(cp_ind[i]+1)
	else:
		tag = str(cp_ind[i]+1)
	
# Load and process recording data
	#spikes, acc, ang, windows = data_load(name, tag, window_length)
	spikes, acc, ang, windows = data_load(name, tag)

# Subject's signals and spike to plot
	sbj_signal = [acc, ang]
	n_spk = 0 #np.random.randint(len(spikes))
	
# Start - stop indices of the recording signal, +/-400 samples arround the selected spike - [wide part]
	strt_wide = int(spikes[n_spk, 0] - 400)
	stp_wide = int(spikes[n_spk, 1] + 400)

# Start - stop indices of the recording signal, strictly arround the spike
	strt_stp = [int(spikes[n_spk, 0]), int(spikes[n_spk, 1])]
	
	plotit(i, sbj_signal)
	

plt.show()
