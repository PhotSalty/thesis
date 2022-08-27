from utils import *

def labeling_samples(dat, k, axs, spks, stp):
	# plot signal with grey background:
	axs[k].plot(np.arange(len(dat)), dat, lw=.5)
	axs[k].set_facecolor('lightgrey')
	axs[k].set_title("Spikes")
	axs[k].grid()
	axs[k].legend(["X","Y","Z"])

	# set labels
	if k == 0:
		axs[k].set_ylabel('g')
	elif k == 1:
		axs[k].set_ylabel('rad')

	axs[k].set_xlabel('samples')
	
	# # define annotation bound
	# if stp == 'raw':
	# 	ab = 2600
	# 	ext = 17
	# elif stp == 'filt':
	# 	ab = 900
	# 	ext = 6.5

	# annotate spikes
	j = 1
	posy = np.min( dat )
	for i in spks:
		# posx = int(np.floor(i[0] + (i[1]-i[0])/2 - 1))
		posx = i[0]
		# axs[k].fill_between(x = i, y1 = -ab*k - ext, y2 = ab*k + ext, color = 'white')
		axs[k].axvspan(xmin = i[0], xmax = i[1], color = 'white')
		axs[k].annotate(str(j), (posx,  posy), color = 'black')
		j = j + 1

	
def plotlab_samples(name, tag, show, stype):

	subject = recording(name, tag)
	subject.read_data()
	subject.filtering()

	spk = subject.spikes
	
	# Initialize signal, based on requested type
	if stype == 'filt':
		acc = subject.filt_acc
		ang = subject.filt_ang
	elif stype == 'raw':
		acc = subject.raw_acc
		ang = subject.raw_ang

	fig, axs = plt.subplots(2)
	fig.suptitle(f'Spikes of Subject <{name}>')
	labeling_samples(acc, 0, axs, spk, stype)
	labeling_samples(ang, 1, axs, spk, stype)

	if show:
		plt.show()



def labeling_seconds(dat, k, axs, spks, stp):
	freq = 64	# Hz
	t = np.arange(len(dat)/freq, step = 1/freq)

	# plot signal with grey background:
	axs[k].plot(t, dat, lw=.5)
	axs[k].set_facecolor('lightgrey')
	axs[k].grid()
	axs[k].legend(["X","Y","Z"])

	# set labels
	if k == 0:
		axs[k].set_ylabel('g')
	elif k == 1:
		axs[k].set_ylabel('rad')

	axs[k].set_xlabel('seconds')
	
	# define annotation bound
	if stp == 'raw':
		ab = 2600
		ext = 17
	elif stp == 'filt':
		ab = 900	
		ext = 6.5

	j = 1
	# annotate spikes
	posy = np.min( dat )
	for i in spks:
		posx = i[0]
		# axs[k].fill_between(x = i, y1 = -ab*k - ext, y2 = ab*k + ext, color = 'white')
		axs[k].axvspan(xmin = i[0], xmax = i[1], color = 'white')
		axs[k].annotate(str(j), (posx,  posy), color = 'black')
		# axs[k].annotate(str(j), (i[0] + (i[1]-i[0])/2 - 1,  -ab*k - ext), color = 'black')
		j = j + 1

	
def plotlab_seconds(name, tag, show, stype):

	subject = recording(name, tag)
	subject.read_data()
	subject.filtering()

	spk = subject.spikes / 64

	# Initialize signal, based on requested type
	if stype == 'filt':
		acc = subject.filt_acc
		ang = subject.filt_ang
	elif stype == 'raw':
		acc = subject.raw_acc
		ang = subject.raw_ang

	fig, axs = plt.subplots(2)
	fig.suptitle(f'Spikes of Subject <{name}>')
	labeling_seconds(acc, 0, axs, spk, stype)
	labeling_seconds(ang, 1, axs, spk, stype)

	if show:
		plt.show()

# names = ['sltn', 'gali', 'sdrf', 'pasx', 'anti', 'komi', 'fot', 'agge', 'conp', 'LH_galios']
# # i = 1
# for i in np.arange(len(names)):
# 	name = names[i]

# 	if i > 8:
# 		tag = str(i+1)
# 	else:
# 		tag = '0' + str(i+1)

# 	plotlab_samples(name, tag)


def subject_selection():
	
	print(f'''\n
> Select a subject from the list:

\t1.  sltn	\t6.  komi

\t2.  gali	\t7.  fot

\t3.  sdrf	\t8.  agge

\t4.  pasx	\t9.  conp

\t5.  anti	\t10. LH_galios\n'''.expandtabs(6) )
        
	subj = int(input(f'  Please, type only the subject\'s number (1-10): '))
	print(' ')
	return subj

flag = False
while not flag:
	subj_number = subject_selection()
	if subj_number >= 1 and subj_number < 10:
		flag = True
		tag = '0' + str(subj_number)
	else:
		clear()
		print(f'\n  Wrong input, please try again.')

# print(f'\n\n{subj_number}')

names = np.asarray(['sltn', 'gali', 'sdrf', 'pasx', 'anti', 'komi', 'fot', 'agge', 'conp', 'LH_galios'])
name = names[subj_number-1]

# if subj_number == 1:
#     name = 'sltn'
# elif subj_number == 2:
#     name = 'gali'
# elif subj_number == 3:
#     name = 'sdrf'
# elif subj_number == 4:
# 	name = 'pasx'
# elif subj_number == 5:
# 	name = 'anti'
# elif subj_number == 6:
# 	name = 'komi'
# elif subj_number == 7:
# 	name = 'fot'
# elif subj_number == 8:
# 	name = 'agge'
# elif subj_number == 9:
# 	name = 'conp'
# elif subj_number == 10:
# 	name = 'LH_galios'


flag = False
while not flag:
	print(f'''
> Signal should be:

\t1. original

\t2. filtered\n'''.expandtabs(6))

	sgntype = str(int(input(f'  Please, type only the signal-type\'s number (1 or 2): ')))
	if sgntype == '1':
		sgntype = 'raw'
		flag = True
	elif sgntype == '2':
		sgntype = 'filt'
		flag = True
	else:
		clear()
		print(f'\n  Wrong input, please try again.')

flag = False
while not flag:
	print(f'''\n
> X-axis should be labeled:

\t1. in seconds

\t2. in samples\n'''.expandtabs(6))

	pltmethod = int(input(f'  Please, type only the plot-method\'s number (1 or 2): '))
	if pltmethod == 1:
		flag = True
		plotlab_seconds(name, tag, show = True, stype = sgntype)
	elif pltmethod == 2:
		flag = True
		plotlab_samples(name, tag, show = True, stype = sgntype)
	else:
		clear()
		print(f'\n  Wrong input, please try again.')

print('')