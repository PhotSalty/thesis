
from utils_case import *
from metacost import MetaCost
from imblearn.over_sampling import SMOTE
from collections import Counter

oversample = SMOTE()

'''
>> We implement the decision-tree with scikit-learn's DecisionTree() method,
   which is using the CART algorithm, instead of C4.5.

>>
'''
# Plotting / printing functions:
def print_trtst_dtree():
        print('\t- LOSO iteration trdata:   '.expandtabs(4), Train8_orig.shape, Train8_auxi.shape, Train8_lbls.shape)
        print('\t- Training data:           '.expandtabs(4), Train8_X.shape, Train8_Y.shape)
        print('\t- LOSO iteration tstdata:  '.expandtabs(4), Tst1_orig.shape, Tst1_auxi.shape, Tst1_lbls.shape)
        print('\t- Testing data:            '.expandtabs(4), Test1_X.shape, Test1_Y.shape)
        print('\t- Full data:               '.expandtabs(4), Tst1_orig.shape[0] + Train8_orig.shape[0], Tst1_auxi.shape[0] + Train8_auxi.shape[0], Tst1_lbls.shape[0] + Train8_lbls.shape[0])


def plot_dtree_predictions():
        plt.figure(f'Subject {i} out')
        plt.plot(Test1_Y, color = 'orange')
        plt.plot(pred, color = 'blue', alpha = 0.5)
        plt.legend(['labels', 'predictions'], loc = 'upper left')
        plt.show()


# 1. Load preprocessed data

pkl_path = 'preprocessed_data' + sls + 'prepdata.pkl'
with open(pkl_path, 'rb') as f:
        s_orig = pkl.load(f)
        s_auxi = pkl.load(f)
        labels = pkl.load(f)
        e_impact = pkl.load(f)
        swing_interval = pkl.load(f)

## Plot auxiliary signal with e-impact:
# m = 10
# print_eimpact(s_auxi, e_impact, m)



# 2. Train Decision Tree - LOSO Validation:

for i in np.arange(len(names)):

# 1. Leaving i-th subject out for total Training

        print(f'\n> Session {i} - <Subject {i}> out\n')

    # Total Test data
        Test_orig_tot = s_orig[i, 0]
        Test_auxi_tot = s_auxi[i, 0]
        Test_lbls_tot = labels[i]

    # Extract testing indicators
        Test_X, Test_Y, Test_ind = extract_indicators(Test_orig_tot, Test_auxi_tot, Test_lbls_tot, e_impact)

    # Training data and spike-labels for 9 subjects
        tr9_orig = s_orig[np.arange(len(s_orig)) != i]
        tr9_auxi = s_auxi[np.arange(len(s_auxi)) != i]
        tr9_lbls = labels[np.arange(len(labels)) != i]

# 2. Decision Tree LOSO Training
        dtree_out_wds  = np.zeros(9, dtype = object)
        dtree_out_lbls = np.zeros(9, dtype = object)
        dtree_out_ind  = np.zeros(9, dtype = object)

        for j in np.arange(len(tr9_orig)):

        # Dtree training data preparation:
                
                # Selecting 8 out of 9 subjects
                Trn8_orig = tr9_orig[np.arange(len(tr9_orig)) != j]
                Trn8_auxi = tr9_auxi[np.arange(len(tr9_auxi)) != j]
                Trn8_lbls = tr9_lbls[np.arange(len(tr9_lbls)) != j]

                # Concatenate training-signals
                Train8_orig = concatenate_lists(Trn8_orig, mde = 1)   # mode 1 -> 3-axes signal
                Train8_auxi = concatenate_lists(Trn8_auxi, mde = 0)   # mode 0 -> 1-axis signal
                Train8_lbls = concatenate_labels(Trn8_lbls)
        
                # Extract training indicators
                Train8_X, Train8_Y, tr8ind = extract_indicators(Train8_orig, Train8_auxi, Train8_lbls, e_impact)


        # Dtree test data preparation:
                
                # Selecting 1 out of 9 subjects
                Tst1_orig = tr9_orig[j, 0]
                Tst1_auxi = tr9_auxi[j, 0]
                Tst1_lbls = tr9_lbls[j]

                # Extract test indicators
                Test1_X, Test1_Y, tst1ind = extract_indicators(Tst1_orig, Tst1_auxi, Tst1_lbls, e_impact)


        # Print training and testing data analysis
                print_trtst_dtree()


        # Perform training via MetaCost algorithm
                alg = tree.DecisionTreeClassifier()
                C = np.array([[0, 1000], [0, 0]])
                S = pd.DataFrame(Train8_X, columns = ['indi1', 'indi2', 'indi3', 'indi4', 'indi5', 'indi6'])
                S['Target'] = Train8_Y
                dtree_model = MetaCost(S, alg, C).fit('Target', 3)


        # Test decision tree with the Test1 (1 subject out of 9 -> Dtree LOSO)
                pred = dtree_model.predict(Test1_X)


        # Plot dtree predictions
                # plot_dtree_predictions()


        # Extract cnn training windows
                '''
                Windows extraction pipeline:

                    tst1ind -> raw signal indices of every sample after thresholding
                    pred    -> for every maxima, decision-tree predicted label

                    When pred == 1, that means that this maxima is a possible relevant
                    movement, so we save the 4second-window and its label.

                    if i-th pred = 1  -->  tstind[i] = raw signal's index of this maxima -.
                                                                                        |
                        .-----------------------------------------------------------------'
                        |
                        '-->  Tst1_orig[ tstind[i] ] = the maxima to extract window from
                '''
                
                # Predicted relevant events indices
                det_ev_ind = tst1ind[np.nonzero(pred)]

                # Windows corresponding to prediction indices
                det_ev_wds, det_ev_lbls, wds_ind = extract_event_windows(det_ev_ind, Tst1_orig, Tst1_lbls)

                # Save data for the cnn LOSO
                dtree_out_wds[j]  = det_ev_wds
                dtree_out_lbls[j] = det_ev_lbls
                dtree_out_ind[j]  = wds_ind     # np.transpose(np.vstack((det_ev_ind - 2*64, det_ev_ind + 2*64)))
                
        
# 3. CNN LOSO training

        for k in np.arange(dtree_out_wds.shape[0]):
        
        # CNN validation data:

                # Selecting 1 out of 9 subjects
                cnn_val1_wds  = dtree_out_wds[k]
                cnn_val1_lbls = dtree_out_lbls[k]
                cnn_val1_ind  = dtree_out_ind[k]

                val_data = (cnn_val1_wds, cnn_val1_lbls)
        
        # CNN training data:

                # Selecting 8 out of 9 subjects
                cnn_trn8_wds  = dtree_out_wds[np.arange(len(dtree_out_wds)) != k]
                cnn_trn8_lbls = dtree_out_lbls[np.arange(len(dtree_out_lbls)) != k]
                cnn_trn8_ind  = dtree_out_ind[np.arange(len(dtree_out_ind)) != k]
                
                # Concatenate windows of all training subjects
                cnn_Train8_X, cnn_Train8_Y, cnn_Train8_ind = concatenate_subj_windows(cnn_trn8_wds, cnn_trn8_lbls, cnn_trn8_ind)

                # Label-balancing through SMOTE resample
                cnn_Train8_X_aug, cnn_Train8_Y_aug = oversample.fit_resample(cnn_Train8_X, cnn_Train8_Y)

                # Print pre and after balance labels
                print(f'\n\tOriginal training data: {Counter(cnn_Train8_Y)}'.expandtabs(4))
                print(f'\tAugmented training data: {Counter(cnn_Train8_Y_aug)}\n'.expandtabs(4))
        
                # DCNN training:
                in_shape = cnn_Train8_X_aug.shape[1:]
                cnn_model, custom_early_stopping = construct_model(in_shape = in_shape)

                history = cnn_model.fit(
                        x = cnn_Train8_X_aug,
                        y = cnn_Train8_Y_aug,
                        batch_size = 200,
                        class_weight = None,
                        validation_data = val_data,
                        verbose = 2,
                        callbacks = [custom_early_stopping]
                )
        
        # Plot model's training evaluation
                fig, axs = plt.subplots(2)
                fig.suptitle(f'Subject out: {i} , Validation subject: {k}')
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
                model_path = sls + 'Output' + sls + 'out' + str(i) + sls + 'Models' + sls + 'M' + str(i) + str(k) +'.mdl'
                figure_path = sls + 'Output' + sls + 'out' + str(i) + sls + 'Training_figures' + sls

                if not os.path.exists(figure_path):
                        os.makedirs(figure_path)

                cnn_model.save(filepath = model_path)

                figure_path += 'M' + str(i) + str(k) + '.pdf'
                fig.savefig(figure_path)


# plt.show()

''' from decision tree's positive predictions, we calculate spikes as a 4 second item '''


