from sklearn.utils import compute_class_weight
from utils import *
from sklearn.model_selection import LeaveOneGroupOut as LOGO

p = os.path.dirname(os.getcwd())
p += sls + 'data' + sls + 'pickle_output' + sls
datapkl = p + 'full_data.pkl'

with open(datapkl, 'rb') as f:
	windows = pkl.load(f)
	labels = pkl.load(f)
	tag = pkl.load(f)
	timestamps = pkl.load(f)
	means = pkl.load(f)
	stds = pkl.load(f)

##### Standardization test - demo:
def check_stnd(labels, windows):

## Windows before and after standardization	
	windows_bfr_stdr = deepcopy(windows)
	windows = apply_stadardization(windows, means, stds)

## Spike-indexes 
	nz = np.transpose(np.nonzero(labels))
	# print(nz.shape)
	q = nz.item(10)

	fig, axs = plt.subplots(2,2)
	axs[0][0].set_title("accelerometer")
	axs[0][0].plot(windows[q, :, 0:3])
	axs[0][0].legend(['a','b','c'])
	axs[1][0].plot(windows_bfr_stdr[q, :, 0:3])
	axs[1][0].legend(['a','b','c'])
	
	axs[0][1].set_title("gyroscope")
	axs[0][1].plot(windows[q, :, 3:])
	axs[0][1].legend(['a','b','c'])
	axs[1][1].plot(windows_bfr_stdr[q, :, 3:])
	axs[1][1].legend(['a','b','c'])
	plt.show()

# check_stnd(labels, windows)


#.-------------------------------------------- Data --------------------------------------------.#
#.																								.#
#.		1. windows																				.# 
#.			- Shape: [num_of_wds, num_of_smpls, num_of_sensors]									.#
#.			- Info:  Processed signal-frames of all subjects									.#
#.																								.#
#.		2. (window) labels																		.#
#.			- Shape: [num_of_wds]																.#
#.			- Info:  Each window, is labeled as "spike" -> 1, "non-spike" -> 0					.#
#.																								.#
#.		3. tag																					.#
#.			- Shape: [num_of_wds]																.#
#.			- Info:  Each subject has it's one tag, in order to identify each windows			.#
#.																								.#
#.		4. timestamps																			.#
#.			- Shape: [num_of_wds]																.#
#.			- Info:  Each window's exact starting-time (refers to the signal file)				.#
#.																								.#
#.----------------------------------------------------------------------------------------------.#


## LOSO method:
def LOSO_training(num_of_epochs, mdl_path):
	
## Subject list by their tags
	subjects = np.unique(tag)
	
	for s in subjects:
		print(f'\n################ Session {s} ################')
		ind = np.asarray(np.where(tag == s))[0]
		print(f'\nTotal number of windows:   {windows.shape[0]} \nSubject-{s} windows:        {ind.shape[0]} \nTrain windows:             {windows.shape[0] - ind.shape[0]}')
		Train_X = windows[:ind[0], :, :]
		Train_Y = labels[:ind[0]]
		print(Train_X.shape, Train_Y.shape)
		Train_X = np.vstack( (Train_X, windows[ind[-1]+1:, :, :]) )
		Train_Y = np.hstack( (Train_Y, labels[ind[-1]+1:]) ).T
		print(Train_X.shape, Train_Y.shape)

		Train_X = apply_stadardization(windows = Train_X, means = means, stds = stds)
		print(f'\n################################ Train Data ready ################################################')
		
	## Input shape: (num_of_samples, num_of_sensors) -> (42, 6)
		in_shape = windows.shape[1:]

	## Model:
		from keras.layers import Dense, MaxPooling1D, LSTM, Dropout, Conv1D, TimeDistributed, BatchNormalization
		from keras.models import Sequential
		from keras.optimizers import RMSprop
		this_optimizer = RMSprop()

		model = Sequential()
		model.add(Conv1D(filters=32, kernel_size=5, padding='same', activation='relu', input_shape=in_shape))
		model.add(BatchNormalization())

		model.add(MaxPooling1D())
		model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
		model.add(BatchNormalization())

		model.add(MaxPooling1D())
		model.add(Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'))
		model.add(BatchNormalization())

		model.add(LSTM(128))
		model.add(Dropout(0.5))
		model.add(Dense(1, activation='sigmoid'))
		
		# model.compile(loss='binary_crossentropy', optimizer='RMSProp', metrics=['accuracy'])
		model.compile(loss='binary_crossentropy', optimizer=this_optimizer, metrics=['accuracy'])
		model.summary()

	## No need to balance class weights, because of balanced data (augmentation)
		# num_of_classes = np.unique(labels)	#[0, 1]
		# weights = compute_class_weight(class_weight='balanced', classes = num_of_classes, y = Train_Y)
	
	## Train model
		model.fit(x=Train_X, y=Train_Y, epochs=num_of_epochs, class_weight=None)
	
	## Save model
		model_path = mdl_path + 'M' + s + '_epochs_' + str(epochs) + '.mdl'
		model.save(filepath=model_path)

epochs = 5
p = os.path.dirname(os.getcwd())
mdl_path = p + sls + 'Models' + sls
# LOSO_training(epochs, mdl_path)
