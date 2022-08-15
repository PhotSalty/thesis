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
Test_orig_tot = np.zeros(len(names), dtype = object)
Test_auxi_tot = np.zeros(len(names), dtype = object)
Test_lbls_tot = np.zeros(len(names), dtype = object)

subj_wds  = np.zeros(len(names), dtype = object)
subj_lbls = np.zeros(len(names), dtype = object)
subj_ind  = np.zeros(len(names), dtype = object)

dtree_model_collection = np.zeros(len(names), dtype = object)

for i in np.arange(len(names)):

# 1. Leaving i-th subject out for total Training

	print(f'\n> Session {i} - <Subject {i}> out\n')

    # Total Test data
	Test_orig_tot[i] = s_orig[i, 0]
	Test_auxi_tot[i] = s_auxi[i, 0]
	Test_lbls_tot[i] = labels[i]

    # Extract testing indicators
	# Test_X, Test_Y, Test_ind = extract_indicators(Test_orig_tot, Test_auxi_tot, Test_lbls_tot, e_impact)

    # Training data and spike-labels for 9 subjects
	tr9_orig = s_orig[np.arange(len(s_orig)) != i, 0]
	tr9_auxi = s_auxi[np.arange(len(s_auxi)) != i, 0]
	tr9_lbls = labels[np.arange(len(labels)) != i]

# 2. Decision Tree LOSO Training
	dtree_out_wds  = np.zeros(9, dtype = object)
	dtree_out_lbls = np.zeros(9, dtype = object)
	dtree_out_ind  = np.zeros(9, dtype = object)

	dt_model_temp = np.zeros(len(names)-1, dtype = object)
	dt_model_path = 'Output' + sls + 'out' + str(i) + sls + 'Models' + sls
	
	if not os.path.exists(dt_model_path):
		os.makedirs(dt_model_path)

	for j in np.arange(len(tr9_orig)):
        
		print(f'\tLOSO iteration {j}'.expandtabs(3))
	
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
		Tst1_orig = tr9_orig[j]
		Tst1_auxi = tr9_auxi[j]
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

		dt_model_temp[j] = dtree_model

	# Test decision tree with the Test1 (1 subject out of 9 -> Dtree LOSO)
		pred = dtree_model.predict(Test1_X)
		
		print(f'\tDecision tree training is over.\n'.expandtabs(3))

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
		
		print(f'\tWindow shape:  {dtree_out_wds[j].shape}, {dtree_out_wds[j][0].shape}'.expandtabs(6))
		print(f'\tLabels shape:  {dtree_out_lbls[j].shape}, {dtree_out_lbls[j][0]}'.expandtabs(6))
		print(f'\tIndices shape: {dtree_out_ind[j].shape}, {dtree_out_ind[j][0]}\n'.expandtabs(6))

	subj_wds[i] = dtree_out_wds
	subj_lbls[i] = dtree_out_lbls
	subj_ind[i] = dtree_out_ind

	dtree_model_collection[i] = dt_model_temp           

# save extractions:
with open('dt_training_data.pkl', 'wb') as f:
	pkl.dump(subj_wds, f)
	pkl.dump(subj_lbls, f)
	pkl.dump(subj_ind, f)

with open('dt_testing_data.pkl', 'wb') as f:
	pkl.dump(Test_orig_tot, f)
	pkl.dump(Test_auxi_tot, f)
	pkl.dump(Test_lbls_tot, f)
	pkl.dump(dtree_model_collection, f)
	pkl.dump(e_impact, f)
