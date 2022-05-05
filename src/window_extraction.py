from filterfuncs import *
import random as rand

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

	# zer = np.zeros( (3, 3) )
	# Q1 = np.vstack( (R, zer) )
	# Q2 = np.vstack( (zer, R) )
	# Q = np.hstack( (Q1, Q2) )
	# print(Q)

	return R

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

def plot_spk_analysis(names):
	spk_time, acc_spk_c, acc_spk_w, ang_spk_c, ang_spk_w = concatenate_signals('all', names)

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

## Plot All-Spikes Analysis:
# plot_spk_analysis()

names = ['sltn', 'sdrf']
spk_len, nspks = concatenate_signals('spk', names)
spikes = pd.DataFrame(spk_len, columns = list(['timelen']))
max_spk_len = round(spikes['timelen'].max(), 3)
## Window_length = maximum spike length of all athlets + offset
## For the case of a better athlete (higher jump), we add a constant offset
jmp_offset = 0.2	# seconds
wd_time_len = max_spk_len + jmp_offset	# seconds
wd_smpl_len = ceil(wd_time_len * 64)	# samples

### Separating into windows:
##  (1) input acc, ang signals and window length
##  (2) iterration with different steps:
##      (i) step = 0.3*wd_smpl_len (samples) -> 0 "spike-window-parts" inside current window
##     (ii) step = 1               (samples) -> N "spike-window-parts" inside current window

name = 'sdrf'
_, spks, _, acc, _, _, ang, _, _ = filtering(name)
s_tot = ceil(len(acc[:, 0]))
## The maximum length of windows = total samples
## Each window will contain wd_smpl_len samples
wds = np.zeros([s_tot, wd_smpl_len, 3])

# acc_temp = np.vstack( ( acc, np.zeros([58, 3]) ) )
# for i in np.arange(s_tot, step = floor(wd_smpl_len*0.3)):
# 	ls = i + wd_smpl_len
# 	# print( np.shape(wds[i,:,:]) , np.shape(acc_temp[i:ls,:]) , (i, len(acc_temp[:,0])))
# 	wds[i, :, :] = acc_temp[i:ls, :]

spks_samp = spks * 64
i = 0
j = 0
s = 0
acc_temp = np.vstack( ( acc, np.zeros([62, 3]) ) )
# print(np.shape(acc_temp), np.shape(acc))

Qz = rand_rot(np.round(np.random.normal(loc=0.0, scale=10, size = None), 3))
loot = []
while s <= s_tot:
	ls = s + wd_smpl_len
	wds[i, :, :] = np.transpose( np.dot(Qz, np.transpose(acc_temp[s:ls, :])))

	## Check_001
	# print((s, ls), np.shape(acc)[0])

	i += 1
	flag = True
	while flag:
		if ls < ceil(spks_samp[j, 0]):
			step = floor(wd_smpl_len*0.3)
			flag = False
		elif s <= ceil(spks_samp[j, 1]):
			step = 1
			flag = False
		else:
			if j < len(spks_samp[:, 0])-1:
				j += 1
			else:
				step = floor(wd_smpl_len*0.3)
				flag = False
	s = s + step
	## Check_002
	loot.append(step)

### Window-extraction loop Checks:
##  (001) Indeces of the last 65-sample saved window.
##        Checking if we cover the full signal length.
##  (002) Step-changes are equal to the number of spikes.
##        We can find a method to label using the step.
## -Scatter loot explanation-
plt.scatter(np.arange(len(loot)), loot)
plt.show()

## Deleting the unused slots in the rear of wds
zer = np.zeros([65, 3])
c = 1
while wds[-c, :, :].any() == zer.any():
	c += 1

## The last non-zero tuple must be equal to the last acc-tuple
## print(wds[-c,:, :], acc[-1, :])

## We exclude the last window, losing a constructed non-spike
## window (mixed signal-tuples with zero-tuples)
wds = wds[1:-c-1, :, :]
print(c)

## -Plot last wds and acc tuples-  **wds are rotated
fig, axs = plt.subplots(2)
axs[0].plot(acc[-66:, :])
axs[1].plot(wds[-1, :, :])
plt.show()

## Labeling: [0, 1]
## (0) -> irrelevant movement
## (1) -> full spike in window

