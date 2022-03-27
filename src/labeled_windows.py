from filterfuncs import *

def plot_spks(acc_spk_cont, acc_spk_wide, ang_spk_cont, ang_spk_wide):
	fig, axs = plt.subplots(2, 2)
	fig.suptitle("Spikes\' signals of 5 athletes", fontweight = 'bold')
	axs[0][0].plot(acc_spk_cont, color = 'darkblue', linewidth = .75)
	axs[0][1].plot(acc_spk_wide, color = 'magenta', linewidth = .75)
	axs[1][0].plot(ang_spk_cont, color = 'darkblue', linewidth = .75)
	axs[1][1].plot(ang_spk_wide, color = 'magenta', linewidth = .75)

	axs[0][0].set_title('Continuous')
	axs[0][1].set_title('Zeros between each spike')

def plot_spk_tmlen(datafr, measurements):
	fig, axs = plt.subplots(2)
	fig.suptitle("Spikes\' time-length of 5 athletes", fontweight = 'bold')
	axs[0].scatter(range(len(datafr['tlngth'])), datafr['tlngth'], color = 'darkgreen', linewidth = .5)
	datafr.plot(kind='hist', ax=axs[1], legend = False, edgecolor = 'white', linewidth = .5)
	datafr.plot(kind='kde', ax=axs[1], secondary_y=True, legend=False, linewidth = 1.5)
	axs[1].set_xlabel('Time (s)')
	axs[1].set_ylabel('Multitude')

	# ax = df.plot(kind='hist')
	# df.plot(kind='kde', ax=ax, secondary_y=True)

	names = ["mode", "median", "mean", "max"]
	colors = ['lightgreen', 'darkblue', 'magenta', 'darkgreen']
	for measurement, name, color in zip(measurements, names, colors):
		plt.axvline(x=measurement, linestyle='--', linewidth=2, label='{0} at {1}'.format(name, measurement), c=color)
	plt.legend()

def plot_spk_analysis():
	spk_time, acc_spk_c, acc_spk_w, ang_spk_c, ang_spk_w = spike_signals('all')

	df = pd.DataFrame(spk_time, columns = list(['tlngth']))
	mode_spk_len = round(df['tlngth'].mode().iloc[0], 2)
	median_spk_len = round(df['tlngth'].median(), 2)
	mean_spk_len = round(df['tlngth'].mean(), 2)
	max_spk_len = round(df['tlngth'].max(), 2)
	measurements = [mode_spk_len, median_spk_len, mean_spk_len, max_spk_len]

	## Plots:
	# (1) Concatenated Spikes:
	plot_spks(acc_spk_c, acc_spk_w, ang_spk_c, ang_spk_w)
	# (2) Time length of Spikes:
	plot_spk_tmlen(df, measurements)

	plt.show()

def find_step(spks_sample, j, wd_smpl_len, s):
	ls = s + wd_smpl_len
	flag = True

	while flag:
		if ls < ceil(spks_sample[j, 0]):
			step = floor(wd_smpl_len*0.3)
			flag = False
		elif s <= ceil(spks_sample[j, 1]):
			step = 1
			flag = False
		else:
			if j < len(spks_sample[:, 0]) -1:
				j+=1
			else:
				step = floor(wd_smpl_len*0.3)
				flag = False
	
	return step

def find_step_and_label(spks_sample, j, wd_smpl_len, s):
	ls = s + wd_smpl_len
	flag = True
	e = round(0.2 * 64)
	while flag:
		bound = ls - round(spks_sample[j, 1])
		if bound < 0:
			if bound > -e:
				step = -bound
			else:
				step = floor(wd_smpl_len*0.3)
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
				step = floor(wd_smpl_len*0.3)
				flag = False
				label = 0

	return step, label

def windowing(name, wd_smpl_len):
	spks, _, acc, _, _, ang, _, _ = filtering(name)
	s_tot = ceil(len(acc[:, 0]))
	## The maximum length of windows = total samples
	## Each window will contain wd_smpl_len samples
	wds = np.zeros([s_tot, wd_smpl_len, 6])
	# wds_acc = np.zeros([s_tot, wd_smpl_len, 3])
	# wds_ang = np.zeros([s_tot, wd_smpl_len, 3])
	labels = wds[:, 0, 0].copy()
	# labels = wds_acc[:, 0, 0]
	
	acc_temp = np.vstack( ( acc, np.zeros([62, 3]) ) )
	ang_temp = np.vstack( ( ang, np.zeros([62, 3]) ) )
	spks_samp = spks * 64
	i = 0
	j = 0
	s = 0

	while s <= s_tot:
		ls = s + wd_smpl_len
		# print(i, s, wd_smpl_len)
		wds[i, :, 0:3] = acc_temp[s:ls, :]
		wds[i, :, 3:] = ang_temp[s:ls, :]
		# wds_acc[i, :, :] = acc_temp[s:ls, :]
		# wds_ang[i, :, :] = ang_temp[s:ls, :]
		step, labels[i] = find_step_and_label(spks_samp, j, wd_smpl_len, s)
		
		i += 1
		s = s + step
		print((s, s_tot, i))

	## Finding the unused slots in the rear of wds
	zer = np.zeros([65, 3])
	c = 1
	while wds[-c, :, :].any() == zer.any():
		c += 1

	## We exclude the last window, losing a constructed non-spike
	## window (mixed signal-tuples with zero-tuples)
	wds = wds[1:-c-1, :, :]
	# wds_acc = wds_acc[1:-c-1, :, :]
	# wds_ang = wds_ang[1:-c-1, :, :]
	labels = labels[1:-c-1]
	return wds, labels, acc, ang
	# return wds_acc, wds_ang, labels, acc, ang


## Plot All-Spikes Analysis:
# plot_spk_analysis()

spikes = pd.DataFrame(spike_signals('spk'), columns = list(['timelen']))
max_spk_len = round(spikes['timelen'].max(), 3)
## Window_length = maximum spike length of all athlets + offset
## For the case of a better athlete (higher jump), we add a constant offset
jmp_offset = 0	# seconds
wd_time_len = max_spk_len + jmp_offset	# seconds
wd_smpl_len = ceil(wd_time_len * 64)	# samples

### Separating into windows:
##  (1) input acc, ang signals and window length
##  (2) iterration with different steps:
##      (i) step = 0.3*wd_smpl_len (samples) -> 0 "spike-window-parts" inside current window
##     (ii) step = 1               (samples) -> N "spike-window-parts" inside current window

wds, labels, acc, ang = windowing('sdrf', wd_smpl_len)
# wds_acc, wds_ang, labels, acc, ang = windowing('sdrf', wd_smpl_len)

## Labels - count and plot:
unique, counts = np.unique(labels, return_counts=True)
d = dict(zip(unique, counts))
print(d)
plt.plot(labels, color = 'teal', linewidth = 1.5)
plt.show()


## -Plot last wds and acc tuples-
# fig, axs = plt.subplots(2,2)
# axs[0][0].plot(acc[-66:, :])
# axs[0][1].plot(wds[-1, :, 0:3])
# axs[1][0].plot(ang[-66:, :])
# axs[1][1].plot(wds[-1, :, 3:])
# plt.show()

