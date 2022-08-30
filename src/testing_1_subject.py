from utils import *
from testing_functions import LOSO_testing, solo_test

def find_label(spikes, j, wdlen, s):

	ls = s + wdlen
	flag = True
	e = round(0.2 * 64)
	ovl = None

	while flag:
		
		bound = ls - spikes[j, 1]

		if bound < 0:
			flag = False
			label = 0
		elif bound <= e:
			flag = False
			label = 1
		else:
			if j < len(spikes[:, 0]) - 1:
				j+=1
			else:
				flag = False
				label = 0
	
	return label, j


def prepare_oversampled_windows(name, tg):
	subj = recording(name = name, tag = f'0{tg}')

	subj.read_data()
	subj.filtering()


	#### Initializations:

	wdlen = 43	# samples

	acc = subj.raw_acc
	ang = subj.raw_ang

	facc = subj.filt_acc
	fang = subj.filt_ang

	spks = subj.spikes


	#### Windowing:

	sub_len = ceil(len(facc[:, 0]))
	wds = np.zeros([sub_len, wdlen, 6])
	labels = deepcopy(wds[:, 0, 0])
	timestamps = deepcopy(labels)

	facc_ext = np.vstack((facc, np.zeros([62, 3])))
	fang_ext = np.vstack((fang, np.zeros([62, 3])))

	i, j, s = 0, 0, 0

	while s <= sub_len:

		ls = s + wdlen

		wds[i, :, :3] = deepcopy(facc_ext[s:ls, :])
		wds[i, :, 3:] = deepcopy(fang_ext[s:ls, :])
		
		timestamps[i] = s / 64

		step = 1
		labels[i], j = find_label(spks, j, wdlen, s)

		i += 1
		s = s + step

	zer = np.zeros([wdlen, 6])
	c = 1
	while np.array_equal(wds[-c, :, :], zer):
		c += 1

	wds = wds[:-c, :, :]
	labels = labels[:-c]
	timestamps = timestamps[:-c]

	subj.windows = deepcopy(wds)
	subj.labels = deepcopy(labels)
	subj.timestamps = deepcopy(timestamps)

	return subj


def perf_test(Test_X, Test_Y, val = 0):

	epochs = 3
	n_subjects = 10

	base_path = os.path.dirname(os.getcwd())

	pkl_path = base_path + sls + 'data' + sls + 'pickle_output' + sls + 'raw_data_' + str(n_subjects) + '_' + file_folder + '.pkl'

	with open(pkl_path, 'rb') as f:
		_ = pkl.load(f)
		_ = pkl.load(f)
		_ = pkl.load(f)
		_ = pkl.load(f)
		means = pkl.load(f)
		stds = pkl.load(f)

	fig_path = base_path + sls + 'new_test' + sls + file_folder + sls + 'epochs_' + str(epochs) 
	mdl_path = base_path + sls + 'Models' + sls + file_folder + sls + 'epochs_' + str(epochs) 
	
	if val == 1:
		mdl_path += '_val-on' + sls
	elif val == 0:
		mdl_path += '_val-off' + sls
		
	return solo_test(Test_X, means, stds, mdl_path, tg)


#### Creating subject

names = np.array(['sltn', 'gali', 'sdrf', 'pasx', 'anti', 'komi', 'fot', 'agge', 'conp', 'LH_galios'])

tg = 2
name = names[tg]

tg = '0' + str(tg)
subject = prepare_oversampled_windows(name, tg)

Test_X = subject.windows
Test_Y = subject.labels
tmstps = subject.timestamps

val = 0

pred_Y = perf_test(Test_X, Test_Y, val = 0)

fig, ax = plt.subplots('Testing oversampled subject')
ax.plot(Test_Y, color = 'blue', linewidth = 3, label = 'Ground-Truth')
ax.plot(pred_Y, color = 'orange', label = 'Prediction')
ax.legend()
ax.grid()

plt.show()