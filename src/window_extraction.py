from filterfuncs import *

spk_time, acc_spk_c, acc_spk_w, ang_spk_c, ang_spk_w = spike_signals('all')

## Plots:

# (1) Concatenated Spikes:
fig, axs = plt.subplots(2, 2)
fig.suptitle("Spikes\' signals of 5 athletes", fontweight = 'bold')
axs[0][0].plot(acc_spk_c, color = 'darkblue', linewidth = .75)
axs[0][1].plot(acc_spk_w, color = 'magenta', linewidth = .75)
axs[1][0].plot(ang_spk_c, color = 'darkblue', linewidth = .75)
axs[1][1].plot(ang_spk_w, color = 'magenta', linewidth = .75)

axs[0][0].set_title('Continuous')
axs[0][1].set_title('Zeros between each spike')


# (2) Time length of Spikes:
df = pd.DataFrame(spk_time, columns = list(['tlngth']))

fig, axs = plt.subplots(2)
fig.suptitle("Spikes\' time-length of 5 athletes", fontweight = 'bold')
axs[0].scatter(range(len(spk_time)), spk_time, color = 'darkgreen', linewidth = .5)
df.plot(kind='hist', ax=axs[1], legend = False, edgecolor = 'white', linewidth = .5)
df.plot(kind='kde', ax=axs[1], secondary_y=True, legend=False, linewidth = 1.5)
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Multitude')


mode_spk_len = round(df['tlngth'].mode().iloc[0], 2)
median_spk_len = round(df['tlngth'].median(), 2)
mean_spk_len = round(df['tlngth'].mean(), 2)
max_spk_len = round(df['tlngth'].max(), 2)

measurements = [mode_spk_len, median_spk_len, mean_spk_len, max_spk_len]
names = ["mode", "median", "mean", "max"]
colors = ['lightgreen', 'darkblue', 'magenta', 'darkgreen']
for measurement, name, color in zip(measurements, names, colors):
    plt.axvline(x=measurement, linestyle='--', linewidth=1.5, label='{0} at {1}'.format(name, measurement), c=color)
plt.legend()

# ax = df.plot(kind='hist')
# df.plot(kind='kde', ax=ax, secondary_y=True)

plt.show()