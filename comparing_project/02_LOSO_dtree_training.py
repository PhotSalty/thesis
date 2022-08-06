
from numpy import extract
from utils_case import *
from metacost import MetaCost

'''
>> We implement the decision-tree with scikit-learn's DecisionTree() method,
   which is using the CART algorithm, instead of C4.5.

>>
'''
                
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

        # Leaving i-th subject out of tree-training
        print(f'\n> Session {i} - <Subject {i}> out\n')

        # Training data and spike-labels for 9 subjects
        tr_orig = s_orig[np.arange(len(s_orig)) != i]
        tr_auxi = s_auxi[np.arange(len(s_auxi)) != i]
        tr_lbls = labels[np.arange(len(labels)) != i]

        # Concatenate training-signals
        Train_orig = concatenate_lists(tr_orig, 1)
        Train_auxi = concatenate_lists(tr_auxi, 0)
        Train_labs = concatenate_labels(tr_lbls)
        
        # Extract training indicators
        Train_X, Train_Y = extract_indicators(Train_orig, Train_auxi, Train_labs, e_impact)

        # Test data
        tst_orig = s_orig[i, 0]
        tst_auxi = s_auxi[i, 0]
        tst_lbls = labels[i]

        # Extract testing indicators
        Test_X, Test_Y = extract_indicators(tst_orig, tst_auxi, tst_lbls, e_impact)

        print('\t- LOSO iteration trdata:   '.expandtabs(4), Train_orig.shape, Train_auxi.shape, Train_labs.shape)
        print('\t- Training data:           '.expandtabs(4), Train_X.shape, Train_Y.shape)
        print('\t- LOSO iteration tstdata:  '.expandtabs(4), tst_orig.shape, tst_auxi.shape, tst_lbls.shape)
        print('\t- Testing data:            '.expandtabs(4), Test_X.shape, Test_Y.shape)
        print('\t- Full data:               '.expandtabs(4), tst_orig.shape[0] + Train_orig.shape[0], tst_auxi.shape[0] + Train_auxi.shape[0], tst_lbls.shape[0] + Train_labs.shape[0])
        

        alg = tree.DecisionTreeClassifier()

        # def metacost_training()
        C = np.array([[0, 1000], [0, 0]])
        S = pd.DataFrame(Train_X, columns = ['indi1', 'indi2', 'indi3', 'indi4', 'indi5', 'indi6'])
        S['Target'] = Train_Y
        model = MetaCost(S, alg, C).fit('Target', 3)
        # model.predict(Test_Y)
        # model.score(S[[0, 1, 2, 3]].values, S['target'])
        # model = model.fit(Train_X, Train_Y)

        prob = model.predict(Test_X)

        plt.figure(f'Subject {i} out')
        plt.plot(Test_Y, color = 'orange')
        plt.plot(prob, color = 'blue', alpha = 0.5)
        plt.legend(['labels', 'predictions'], loc = 'upper left')
        # plt.show()

plt.show()

'''
        Isws sto Preprocessing, tha prepei na etoimasw:

                - s_origin, s_auxiliary
                - labels according to spikes
                - e_impact according to spikes mean amplitude ~ or mb same
                - swing-interval in samples
        
        kai se kathe LOSO iteration na exw etoima signal_samples + sample_labels 
        gia to DecisionTree training, alla kai gia to testing
        
'''
# events = np.empty((10, 2), dtype = object)
# ex = e_impact.iloc[0, 0]
# ey = e_impact.iloc[0, 1]
# ez = e_impact.iloc[0, 2]

# for s, so, pos in zip(s_auxi, s_orig, np.arange(len(s_orig))):

#       for i in np.arange(s.shape[0]):
                
#               # current subject events
#               temp_events = []
        
#               # auxiliary signal values
#               x = np.abs(s[i, 0])
#               y = np.abs(s[i, 1])
#               z = np.abs(s[i, 2])
                
#               # original signal values
#               xo = np.abs(so[i, 0])
#               yo = np.abs(so[i, 1])
#               zo = np.abs(so[i, 2])
                

#               if x >= ex and y >= ey and z >= ez:
                        
#                       # Amplitude indicator
#                       ampl_ind = np.asarray([xo, yo, zo])
                        
#                       # Swing movement average indicator
#                       swing_mv = np.asarray(np.arange(start = i - swing_interval, stop = i))
#                       swng_ind = np.asarray(np.mean(so[swing_mv]))

# #             
# #  flag = decision_tree(ind1 = ampl_ind, ind2 = swng_ind)
# #
# #  if flag:
# #  
# #      temp_events.append(i) -> append position of event
# #
                
#       events[pos, 0] = temp_events
#       events[pos, 1] = names[pos]


# '''
#       > Scikit-learn's DecisionTree() algorithm will be used.
                
#               - It is based on CART algorithm, which is quite similar
#                 to C4.5, which is proposed
# '''
