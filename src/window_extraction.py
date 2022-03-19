from matplotlib.pyplot import legend
from filterfuncs import *
import pandas as pd

## Initialization:
# (1) Input:
names = np.array(["gali", "sdrf", "sltn", "pasx", "anti"])
spk_dif = list()

acc_spk_c = list()
acc_spk_w = list()
ang_spk_c = list()
ang_spk_w = list()

#return order: spk_dif, facc, acc_spk_cont, acc_spk_wide, fang, ang_spk_cont, ang_spk_wide
for name in names:
	dif, _, a, b, _, c, d = filtering(name)
	spk_dif.extend(dif)
	acc_spk_c.extend(a)
	acc_spk_w.extend(b)
	ang_spk_c.extend(c)
	ang_spk_w.extend(d)

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
df = pd.DataFrame(spk_dif, columns = list(['tlngth']))

fig, axs = plt.subplots(2)
fig.suptitle("Spikes\' time-length of 5 athletes", fontweight = 'bold')
axs[0].scatter(range(len(spk_dif)), spk_dif, color = 'darkgreen', linewidth = .5)
df.plot(kind='hist', ax=axs[1], legend = False, edgecolor = 'white', linewidth = .5)
df.plot(kind='kde', ax=axs[1], secondary_y=True, legend = False, linewidth = 1.5)
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Multitude')

# ax = df.plot(kind='hist')
# df.plot(kind='kde', ax=ax, secondary_y=True)

plt.show()