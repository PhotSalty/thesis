
from utils_case import *

# Paths:
base_path = os.path.dirname(os.getcwd())
rec_path = base_path + sls + 'data' + sls + 'recordings' + sls
spk_path = base_path + sls + 'data' + sls + 'pickled_data' + sls + file_folder + sls

# Define arrays to use:
s_orig = np.empty((10, 2), dtype = object)
s_auxi = np.empty((10, 2), dtype = object)
spikes_exact = np.empty((10, 2), dtype = object)
labels = np.empty((10), dtype = object)
e_impact = np.float32(100)
# mxsum = np.zeros((10, 3), dtype = np.float16)
# mnsum = np.zeros((10, 3), dtype = np.float16)


# 2. Process and save signals for each subject

# Iterate through indices, names and tags
for i, nm, tg in zip(np.arange(len(names)), names, tags):

	# Load original signal
	acc_path = rec_path + nm + 'acc.bin'
	acc = np.fromfile(acc_path, dtype = np.int16, sep = '')
	acc = acc.reshape(len(acc)//3, 3)
	acc = acc * 0.0004

	if not (nm.find('LH') == -1):
		acc_x = np.array([
			[-1, 0, 0],
			[0, 1, 0],
			[0, 0, 1]])

		acc = acc @ acc_x
		print(f'\n  > {i}th subject is left handed')
  

# Extract auxiliary signal

	# High-pass filtering
	hp_acc = np.asarray(deepcopy(acc), dtype = np.int64)
	for j in np.arange(hp_acc.shape[1]):
		hp_acc[:, j] = hl_butter_filter(hp_acc[:, j], 8, 'high')
	
	# L1-norm of the high-passed signal
	l1_facc = np.zeros(hp_acc[:, 0].shape)
	for j in np.arange(len(l1_facc)):
		l1_facc[j] = np.sum(np.abs(hp_acc[j, :]))

	# Low-pass filtering
	facc = hl_butter_filter(l1_facc, 3, 'low')

	# Save signals
	s_orig[i, 0] = acc
	s_auxi[i, 0] = facc
	s_orig[i, 1] = nm
	s_auxi[i, 1] = nm

	# Load spikes
	with open(spk_path + nm + '.pkl', 'rb') as f:
		data = pkl.load(f)
	
	# Fully annotated spikes
	spikes_annotated = data[0]

	# Convert spikes from seconds to samples
	spikes_annotated = np.asarray(np.ceil(np.asarray(spikes_annotated * FS)), dtype = np.int64)
	
	'''
		Spikes are annotated in four spots:

			0. Arm-swinging (approach)
                                                   
			1. Last ground-touch (before jump)    <-----.   /
                                                        | \/
			2. First ground-touch (after jump)    <-----'

			3. Both feet on the ground (almost stable)
		
		Search for the maximum amplitude in the auxiliary signal 
		between spike-steps (1) and (2), in order to determine
		the exact spike time
	
	'''
# Exact spike-samples calculation
	exspks = []
	for s in spikes_annotated:
		# index of abs-max of auxi-signal
		temp = np.argmax(np.abs(facc[s[1]:s[2]]))
		# in the case of multiple indices, keep the first one
		if not np.isscalar(temp):
			temp = temp[0]
		# append the spike-index in the subject's list
		temp += s[1]
		exspks.append(temp)

	# Save spike indices
	spikes_exact[i, 0] = np.asarray(exspks, dtype = np.int64)
	spikes_exact[i, 1] = nm

	spk_ind = spikes_exact[i, 0]
	n_samples = len(s_orig[i, 0][:, 0])
	
	# Number of labels = number of samples
	labels[i] = np.zeros( n_samples, dtype = np.int16 )			# non-spikes label = 0
	labels[i][spk_ind] = 1										# spikes-label     = 1

# e-impact selection
	auxi_spks = s_auxi[i, 0][spk_ind]
	min_spk_acc = np.min(np.abs(auxi_spks))

	if e_impact > min_spk_acc:
		e_impact = min_spk_acc

# e-impact is selected as the 20% of the minimum spike amplitude:
e_impact = np.float16(np.round(0.05 * e_impact, 5))
print(f'\n  > Selected e-impact: {e_impact:.3f}\n')

# Basic attributes ready:
# e_impact = pd.DataFrame( temp, columns = ['x', 'y', 'z'], index = [''] )
swing_interval = ceil(0.2 * FS)  # 200 ms -> samples


'''
	Preprocessed data:

		1. Original signals  x10       ->    s_orig[10, 2]    (0 object-cell signal, 1 object-cell name of each subject)

		2. Auxiliary signals x10       ->    s_auxi[10, 2]    (0 object-cell signal, 1 object-cell name of each subject)

		3. Sample labels			   ->    labels[10]       (For each recorded sample, if (spike) label = 1; else label = 0)

		4. e-impact of x, y, z axes    ->    e_impact         (L1-norm of minimum amplitude spike of all subjects)

		5. Swing-interval in samples   ->    swing_interval   (200 ms interval converted to samples)

		6. Exact spike-time sample     ->    spikes_exact     (spike-sample indeces)

'''

# Save data:
pkl_path = 'preprocessed_data' + sls

if not os.path.exists(pkl_path):
	os.makedirs(pkl_path)

with open(pkl_path + 'prepdata.pkl', 'wb') as f:
	pkl.dump(s_orig, f)
	pkl.dump(s_auxi, f)
	pkl.dump(labels, f)
	pkl.dump(e_impact, f)
	pkl.dump(swing_interval, f)

