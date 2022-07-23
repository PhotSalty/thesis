
from utils import *
from training_functions import LOSO_training_noval, LOSO_training_val
from testing_functions import LOSO_testing

## Initializations

# 1. Save given epochs and number of used subjects
epochs, n_subjects = np.int16(sys.argv[1]), np.int16(sys.argv[2])

# 2. Base folder path
base_path = os.path.dirname(os.getcwd())

# 3. Construct pickle-file path of preprocessed data
pkl_path = base_path + sls + 'data' + sls + 'pickle_output' + sls


## Model Training

# 1. Retrieve augmented data
pkl_path += 'aug_data_' + str(n_subjects) + '_' + file_folder + '.pkl'
with open(pkl_path, 'rb') as f:
        windows = pkl.load(f)
        labels = pkl.load(f)
        tag = pkl.load(f)
        timestamps = pkl.load(f)
        means = pkl.load(f)
        stds = pkl.load(f)

# 2. Enable or disable validation
flag = False
while not flag:
	print(f'''
> How do you want to perform training:

\t1. with validation

\t2. without validation
''')
	val_check = int(input(f' Please, type only the training-method\'s number (1 or 2)'))
	if val_check == 1:
		flag = True
		val_check = True
	elif val_check == 2:
		flag = True
		val_check = False
	else:
		clear()
		print(f'\n  Wrong input, please try again.')

# 3. Construct Model-save path
mdl_path = base_path + sls + 'Models' + sls + file_folder + sls + 'epochs_' + str(epochs) 

# 4. Perform Training and save models
if val_check:
	mdl_path += '_val-on' + sls
	val_list = LOSO_training_val(windows, labels, tag, means, stds, epochs, n_subjects, mdl_path)
else:
	mdl_path += '_val-off' + sls
	LOSO_training_noval(windows, labels, tag, means, stds, epochs, n_subjects, mdl_path)


## Model Testing

# 1. Retrieve original - raw data
pkl_path = base_path + sls + 'data' + sls + 'pickle_output' + sls + 'raw_data_' + str(n_subjects) + '_' + file_folder + '.pkl'
with open(pkl_path, 'rb') as f:
        windows = pkl.load(f)
        labels = pkl.load(f)
        tag = pkl.load(f)
        timestamps = pkl.load(f)
        means = pkl.load(f)
        stds = pkl.load(f)

# 2. Perform training, display results and save prediction plots
LOSO_testing(windows, labels, tag, means, stds, epochs, n_subjects, mdl_path)



