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

## Plot All-Spikes Analysis:
# plot_spk_analysis()

spikes = pd.DataFrame(spike_signals('spk'), columns = list(['timelen']))
max_spk_len = round(spikes['timelen'].max(), 3)
# Window_length = maximum spike length of all athlets + offset
# For the case of a better athlete (higher jump), we add a constant offset
jmp_offset = 0.2	# seconds
wd_time_len = max_spk_len + jmp_offset	# seconds
wd_smpl_len = wd_time_len * 64	# samples

## Separating into windows:
# (1) input acc, ang signals and window length
# (2) iterration with different steps:
#     (i) step = 0.3*wd_smpl_len (samples) -> 0 "spike-window-parts" inside current window
#    (ii) step = 1               (samples) -> N "spike-window-parts" inside current window