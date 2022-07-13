from sklearn.utils import compute_class_weight
from utils import *
from sklearn.model_selection import LeaveOneGroupOut as LOGO
from keras.layers import Dense, MaxPooling1D, LSTM, Dropout, Conv1D, TimeDistributed, BatchNormalization
from keras.models import Sequential, save_model
from keras.optimizers import RMSprop
import tensorflow as ts

p = os.path.dirname(os.getcwd())
p1 = p + sls + 'data' + sls + 'pickle_output' + sls
datapkl = p1 + 'aug_data10.pkl'

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


##################################. Training Data analysis .####################################
#.                                                                                            .#
#.      1. windows                                                                                .#
#.         - Shape: [num_of_wds, num_of_smpls, num_of_sensors]                                    .#
#.         - Info:  Processed signal-frames of all subjects                                       .#
#.                                                                                            .#
#.      2. (window) labels                                                                        .#
#.         - Shape: [num_of_wds]                                                                  .#
#.         - Info:  Each window, is labeled as "spike" -> 1, "non-spike" -> 0                     .#
#.                                                                                            .#
#.      3. tag                                                                                    .#
#.         - Shape: [num_of_wds]                                                                  .#
#.         - Info:  Each subject has it's one tag, in order to identify each windows              .#
#.                                                                                            .#
#.      4. timestamps                                                                             .#
#.         - Shape: [num_of_wds]                                                                  .#
#.         - Info:  Each window's exact starting-time (refers to the signal file)                 .#
#.                                                                                            .#
################################################################################################


def construct_training_set(windows, labels, ind_low, ind_high):
##                        ind1           ind2            ind3
## in this case, we have [...., ind_low, ...., ind_high, ....]
        ind1 = np.arange(start = 0, stop = ind_low[0])
        ind2 = np.arange(start = ind_low[-1]+1, stop = ind_high[0])
        ind3 = np.arange(start = ind_high[-1]+1, stop = windows.shape[0])

## Constructing X:
        first_part = windows[ind1, :, :]
        second_part = windows[ind2, :, :]
        third_part = windows[ind3, :, :]
        Train_X = np.vstack((first_part, second_part, third_part))

## Constructing Y:
        first_part = labels[ind1]
        second_part = labels[ind2]
        third_part = labels[ind3]
        Train_Y = np.hstack((first_part, second_part, third_part))
        
        return Train_X, Train_Y

## LOSO method:
def LOSO_training(num_of_epochs, mdl_path):
        
## Subject list by their tags
        subjects = np.unique(tag)
        val_list = np.array([])
        
        for s in subjects:
        
                print(f'\n################ Session {s} ################')
                
## Validation Data:
                print(f'\nGenerating Validation Data:')
                
        ## validation subject != training subject
                flag = True
                while flag:
                        rnd = np.random.randint(0, np.shape(subjects)[0]-1)
                        s_val = subjects[rnd]
                        if s_val != s:
                                flag = False

        ## Save a list of subjects selected for validation in each session
                val_list = np.append(val_list, s_val)
                
        ## Validation subject's indices
                ind_val = np.asarray(np.where(tag == s_val))[0]

        ## Constructing Validation window-set
                Val_X = windows[ind_val[0]:ind_val[-1]+1, :, :]
                Val_Y = labels[ind_val[0]:ind_val[-1]+1]
                Val_X = apply_stadardization(windows = Val_X, means = means, stds = stds)
  
                Val_data = (Val_X, Val_Y)
                print(f'\n\tValidation Subject: {s_val}\n\tValidation Data: {Val_X.shape[0]}')


## Training Data:
                print(f'\nGenerating Training Data:')
        
        ## Training subject's indices
                ind_trn = np.asarray(np.where(tag == s))[0]

                if ind_trn[0] < ind_val[0]:
                        Train_X, Train_Y = construct_training_set(windows = windows, labels = labels, ind_low = ind_trn, ind_high = ind_val)    
                else:
                        Train_X, Train_Y = construct_training_set(windows = windows, labels = labels, ind_low = ind_val, ind_high = ind_trn)
                
                Train_X = apply_stadardization(windows = Train_X, means = means, stds = stds)
                
                print(f'\n\tTesting Subject: {s}\n\tTesting Data: {ind_trn.shape[0]}')
                
                print(f'\nTotal number of windows:         {windows.shape[0]}')
                print(f'Testing Subject-{s} windows:      {ind_trn.shape[0]}')
                print(f'Validation Subject-{s_val} windows:   {ind_val.shape[0]}')
                print(f'Training data windows:           {windows.shape[0] - ind_trn.shape[0] - ind_val.shape[0]}')
                print(f'Extracted training data:         {Train_X.shape[0]}')
        
        ## Input shape: (num_of_samples, num_of_sensors) -> (42, 6)
                in_shape = windows.shape[1:]

        ## Model:
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
                # num_of_classes = np.unique(labels)    #[0, 1]
                # weights = compute_class_weight(class_weight='balanced', classes = num_of_classes, y = Train_Y)
        
        ## Train model
                history = model.fit(x=Train_X, y=Train_Y, epochs=num_of_epochs, verbose = 2, class_weight=None, validation_data = Val_data)

                fig, axs = plt.subplots(2)
                fig.suptitle(f'Subject out: {s}')
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

        ## Save model
                model_path = mdl_path + 'LOSO_' + subjects[-1] # str(subjects.shape[0])
                figure_path = model_path + sls + 'figures'

                if not os.path.exists(figure_path):
                        os.makedirs(figure_path)                        # figure_path contains model_path

                model_path += sls + 'M' + s + '_epochs_' + str(epochs) + '.mdl'
                model.save(filepath=model_path)

                figure_path += sls + 'M' + s + '.pdf'
                fig.savefig(figure_path)

        return val_list

epochs = np.int16(sys.argv[1])
#epochs = 10

mdl_path = p + sls + 'Models' + sls + 'epochs_' + str(epochs) + sls
val_list = LOSO_training(epochs, mdl_path)

print(val_list)
plt.show()
