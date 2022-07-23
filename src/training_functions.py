
from utils import *

from sklearn.utils import compute_class_weight
from sklearn.model_selection import LeaveOneGroupOut as LOGO
from keras.layers import Dense, MaxPooling1D, LSTM, Dropout, Conv1D, TimeDistributed, BatchNormalization
from keras.models import Sequential
from keras.optimizers import RMSprop
# import tensorflow as ts



###############################. Training Data analysis .###################################
##                                                                                        ##
##   1. windows                                                                           ##
##      - Shape: [num_of_wds, num_of_smpls, num_of_sensors]                               ##
##      - Info:  Processed signal-frames of all subjects                                  ##
##                                                                                        ##
##   2. (window) labels                                                                   ##
##      - Shape: [num_of_wds]                                                             ##
##      - Info:  Each window, is labeled as "spike" -> 1, "non-spike" -> 0                ##
##                                                                                        ##
##   3. tag                                                                               ##
##      - Shape: [num_of_wds]                                                             ##
##      - Info:  Each subject has one specific tag, in order to identify its windows      ##
##                                                                                        ##
##   4. timestamps                                                                        ##
##      - Shape: [num_of_wds]                                                             ##
##      - Info:  Each window's exact starting-time (refers to the signal file)            ##
##                                                                                        ##
############################################################################################


######  TRAINING MODEL STRUCTURE  ######################################################################

def training_model(in_shape):
	
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
	
	model.compile(loss='binary_crossentropy', optimizer='RMSProp', metrics=['accuracy'])
	# model.summary()

	return model

########################################################################################### END ######




######  TRAINING WITH VALIDATION  #################################################################### 


###  1. Separate Training windows from Validation and Testing windows #######################

def construct_training_set_val(windows, labels, ind_low, ind_high):

# Windows Structure:
#
#   ind1            ind2             ind3
# [ .... , ind_low, .... , ind_high, .... ]
#   
# So, training data will be those in ind 1,2,3	
	ind1 = np.arange(start = 0, stop = ind_low[0])
	ind2 = np.arange(start = ind_low[-1]+1, stop = ind_high[0])
	ind3 = np.arange(start = ind_high[-1]+1, stop = windows.shape[0])

# Constructing X - windows
	first_part = windows[ind1, :, :]
	second_part = windows[ind2, :, :]
	third_part = windows[ind3, :, :]
	Train_X = np.vstack((first_part, second_part, third_part))

# Constructing Y - labels
	first_part = labels[ind1]
	second_part = labels[ind2]
	third_part = labels[ind3]
	Train_Y = np.hstack((first_part, second_part, third_part))
        
	return Train_X, Train_Y


#################################################################################  END 1  ###


###  2. Perform LOSO training ###############################################################

def LOSO_training_val(windows, labels, tag, means, stds, epochs, n_subjects, mdl_path):

	print(f'\n>> Training with validation')

# List of available subject tags
	subject_tags = np.unique(tag)

# List to save subjects used as "validation set" in each session
	val_list = np.array([])


# LOSO iterations through subjects
	for st in subject_tags:

		print(f'\n > Session {st}: leaving subject <{st}> out')

	# Validation and testing subject must be different
		flag = True
		while flag:
			rnd = np.random.randint(0, np.shape(subject_tags)[0])
			s_val = subject_tags[rnd]

			if s_val != st:
				flag = False

	# Update validation list
		val_list = np.append(val_list, s_val)

	# Find validation subject's indices
		ind_val = np.asarray(np.where(tag == s_val))[0]

	# Extract validation windows and their labels
		Val_X = windows[ind_val[0]:ind_val[-1]+1, :, :]
		Val_Y = labels[ind_val[0]:ind_val[-1]+1]
		Val_X = apply_standardization(windows = Val_X, means = means, stds = stds)

		Val_data = (Val_X, Val_Y)

	# Find training subject's indices
		ind_trn = np.asarray(np.where(tag == st))[0]

	# Extract training windows and their labels
		if ind_trn[0] < ind_val[0]:
			Train_X, Train_Y = construct_training_set_val(windows = windows, labels = labels, ind_low = ind_trn, ind_high = ind_val)    
		else:
			Train_X, Train_Y = construct_training_set_val(windows = windows, labels = labels, ind_low = ind_val, ind_high = ind_trn)
		
		Train_X = apply_standardization(windows = Train_X, means = means, stds = stds)

	# Display data separation
		print(f'''
\t           Total number of windows    ->    {windows.shape[0]}
\t        Testing Subject-{st} windows    ->    {ind_trn.shape[0]}
\t     Validation Subject-{s_val} windows    ->    {ind_val.shape[0]}
\t  Calculated training data windows    ->    {windows.shape[0] - ind_trn.shape[0] - ind_val.shape[0]}
\t   Extracted training data windows    ->    {Train_X.shape[0]}

		'''.expandtabs(4))

	# Define input shape: (num_of_samples, num_of_sensors) -> (window_length ~ 41, 6)
		in_shape = windows.shape[1:]

	# Get model
		model = training_model(in_shape)

	# No need to balace class weights, because of preprocessing balance (artificial augmentation)
		# num_of_classes = np.unique(labels)    #[0, 1]
		# weights = compute_class_weight(class_weight='balanced', classes = num_of_classes, y = Train_Y)

	# Train model
		history = model.fit(x = Train_X, y = Train_Y, epochs = epochs, verbose = 2, class_weight = None, validation_data = Val_data)

	# Plot model's training evaluation
		fig, axs = plt.subplots(2)
		fig.suptitle(f'Subject out: {st} , Validation subject: {s_val}')
		#  "Accuracy"
		axs[0].plot(history.history['accuracy'])
		axs[0].plot(history.history['val_accuracy'])
		axs[0].set_title('Model accuracy')
		axs[0].set_ylabel('accuracy')
		axs[0].set_xlabel('epoch')
		axs[0].legend(['train', 'validation'])
		# "Loss"
		axs[1].plot(history.history['loss'])
		axs[1].plot(history.history['val_loss'])
		axs[1].set_title('Model loss')
		axs[1].set_ylabel('loss')
		axs[1].set_xlabel('epoch')
		axs[1].legend(['train', 'validation'])

	# Save model and its figures
		model_path = mdl_path + 'M' + st + '_' + str(n_subjects) + '.mdl'
		figure_path = mdl_path + 'Training_figures' + sls

		if not os.path.exists(figure_path):
			os.makedirs(figure_path)

		model.save(filepath = model_path)

		figure_path += 'M' + st + '_' + str(n_subjects) + '.pdf'
		fig.savefig(figure_path)

	return val_list

##################################################################################  END 2  #######


#########################################################################################################




######  TRAINING WITHOUT VALIDATION  #################################################################### 


###  1. Separate Training from Testing windows ##################################################

def construct_training_set_noval(windows, labels, ind):

# Windows Structure:
#
#   ind1         ind2  
# [ .... , ind , .... ]
#   
# So, training data will be those in ind 1,2	
	ind1 = np.arange(start = 0, stop = ind[0])
	ind2 = np.arange(start = ind[-1]+1, stop = windows.shape[0])

# Constructing X - windows
	first_part = windows[ind1, :, :]
	second_part = windows[ind2, :, :]
	Train_X = np.vstack((first_part, second_part))

# Constructing Y - labels
	first_part = labels[ind1]
	second_part = labels[ind2]
	Train_Y = np.hstack((first_part, second_part))
        
	return Train_X, Train_Y


#################################################################################  END 1  ###


###  2. Perform LOSO training ###############################################################

def LOSO_training_noval(windows, labels, tag, means, stds, epochs, n_subjects, mdl_path):

	print(f'\n>> Training without validation')

# List of available subject tags
	subject_tags = np.unique(tag)

# LOSO iterations through subjects
	for st in subject_tags:

		print(f'\n > Session {st}: leaving subject <{st}> out')

	# Find training subject's indices
		ind_trn = np.asarray(np.where(tag == st))[0]

	# Extract training windows and their labels
		Train_X, Train_Y = construct_training_set_noval(windows, labels, ind_trn)    
		
		Train_X = apply_standardization(windows = Train_X, means = means, stds = stds)

	# Display data separation
		print(f'''
\t           Total number of windows    ->    {windows.shape[0]}
\t        Testing Subject-{st} windows    ->    {ind_trn.shape[0]}
\t  Calculated training data windows    ->    {windows.shape[0] - ind_trn.shape[0]}
\t   Extracted training data windows    ->    {Train_X.shape[0]}

		'''.expandtabs(4))

	# Define input shape: (num_of_samples, num_of_sensors) -> (window_length ~ 41, 6)
		in_shape = windows.shape[1:]

	# Get model
		model = training_model(in_shape)

	# No need to balace class weights, because of preprocessing balance (artificial augmentation)
		# num_of_classes = np.unique(labels)    #[0, 1]
		# weights = compute_class_weight(class_weight='balanced', classes = num_of_classes, y = Train_Y)

	# Train model
		history = model.fit(x = Train_X, y = Train_Y, epochs = epochs, verbose = 2, class_weight = None)

	# Plot model's training evaluation
		fig, axs = plt.subplots(2)
		fig.suptitle(f'Subject out: {st}')
		#  "Accuracy"
		axs[0].plot(history.history['accuracy'])
		axs[0].set_title('Model accuracy')
		axs[0].set_ylabel('accuracy')
		axs[0].set_xlabel('epoch')
		axs[0].legend(['train'])
		# "Loss"
		axs[1].plot(history.history['loss'])
		axs[1].set_title('Model loss')
		axs[1].set_ylabel('loss')
		axs[1].set_xlabel('epoch')
		axs[1].legend(['train'])

	# Save model and its figures
		model_path = mdl_path + 'M' + st + '_' + str(n_subjects) + '.mdl'
		figure_path = mdl_path + 'Training_figures' + sls

		if not os.path.exists(figure_path):
			os.makedirs(figure_path)

		model.save(filepath = model_path)

		figure_path += 'M' + st + '_' + str(n_subjects) + '.pdf'
		fig.savefig(figure_path)


##################################################################################  END 2  #######


############################################################################################################
