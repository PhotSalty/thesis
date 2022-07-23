
from utils import *

names = np.array(['sltn', 'gali', 'sdrf', 'pasx', 'anti', 'komi', 'fot', 'agge', 'conp', 'LH_galios'])


## Subject data processing

# 1. Initialization
subjects, ns, po, ne = subjects_init(names)
ns = np.array(ns) - 1

# 2. Balance windows - Artificial augmentation of positive-class windows"
subjects_aug = balance_windows(subjects, ns, po, ne)

# * Display base's statistical analysis - [OPTIONAL]
display_base_stats(subjects)


## Extract training data

# 1. Using subjects with original - raw windows
wds_raw, lbl_raw, tg_raw, tmst_raw = concatenate_windows(subjects)
means_raw, stds_raw = standardization_parameters(wds_raw)

# 2. Using subjects with balanced windows
wds_aug, lbl_aug, tg_aug, tmst_aug = concatenate_windows(subjects_aug)
means_aug, stds_aug = standardization_parameters(wds_aug)

# * Standardization parameters differ in those two approaches, so we need to select
#   which "package" to use:
means, stds = means_raw, stds_raw


## Constructing data pickles

# 1. Define pickled-data path
base_folder = os.path.dirname(os.getcwd())
pkl_folder  = base_folder + sls + 'data' + sls + 'pickle_output' + sls

data_raw_path = pkl_folder + 'raw_data_' + str(len(names)) + '_' + file_folder + '.pkl'
data_aug_path = pkl_folder + 'aug_data_' + str(len(names)) + '_' + file_folder + '.pkl'

# 2. Save pickles
with open(data_raw_path, 'wb') as f:
	pkl.dump(wds_raw, f)
	pkl.dump(lbl_raw, f)
	pkl.dump(tg_raw, f)
	pkl.dump(tmst_raw, f)
	pkl.dump(means, f)
	pkl.dump(stds, f)

with open(data_aug_path, 'wb') as f:
	pkl.dump(wds_aug, f)
	pkl.dump(lbl_aug, f)
	pkl.dump(tg_aug, f)
	pkl.dump(tmst_aug, f)
	pkl.dump(means, f)
	pkl.dump(stds, f)


