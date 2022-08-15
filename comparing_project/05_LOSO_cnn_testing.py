
from utils_case import *
from keras.models import load_model
from sklearn.metrics import confusion_matrix

with open('dt_testing_data.pkl', 'rb') as f:
    tst_orig = pkl.load(f)
    tst_auxi = pkl.load(f)
    tst_lbls = pkl.load(f)
    dtree_model_collection = pkl.load(f)
    e_impact = pkl.load(f)

cm_tot = []

fig_path = 'Output' + sls + 'Testing_figures' + sls
    
if not os.path.exists(fig_path):
    os.makedirs(fig_path)

for s in np.arange(10):

    # Load s-subject data
    dtree_s = dtree_model_collection[s]
    Test_orig = tst_orig[s]
    Test_auxi = tst_auxi[s]
    Test_lbls = tst_lbls[s]

    # Confusion matrix of s-subject trials
    cm_s = []

    # Subject-s testing figures
    fig = np.zeros(10, dtype=object)
    fi = 0

    # Cnn models inital path
    cnn_path_s = 'Output' + sls + 'out' + str(s) + sls + 'Models' + sls
    
    # Extract s-subject indicators for dtree
    Test_X, Test_Y, Test_ind = extract_indicators(Test_orig, Test_auxi, Test_lbls, e_impact)
    
    print(f'> Testing subject: {s}\n')

    for i in np.arange(9):
        
        print(f' - Using models {s}_{i}...\n'.expandtabs(4))
        # dtree s_i model
        dtree_s_i = dtree_s[i]

        # load cnn s_i model
        cnn_path_s_i = cnn_path_s + 'M_' + str(s) + '_' + str(i) + '.mdl'
        cnn_s_i = load_model(cnn_path_s_i)

        pred = dtree_s_i.predict(Test_X)

        # predicted relevant event indices
        detected_events_ind = Test_ind[np.nonzero(pred)]

        # Windows corresponding to prediction indices
        de_wds, de_lbls, de_ind = extract_event_windows(detected_events_ind, Test_orig, Test_lbls)

        #print(de_wds.shape, de_lbls.shape)

        cnn_X, cnn_Y = de_wds, de_lbls
        cnn_pred = cnn_s_i.predict(x = cnn_X)

        # Complete prediction with pre-calculated threshold -> 0.5
        cnn_pred_Y = np.where(cnn_pred < 0.5, 0, 1)
        
        # Total prediction init
        total_pred_Y = deepcopy(pred)
        
        # k-index runs through cnn predictions. length( Cnn_predictions ) = length( non-zeros(tree-pred) )
        k = 0

        for j in np.arange(len(pred)):

            if pred[j] == 1:
            
            # if the corresponding window's cnn-prediction is 0, change total prediction
                if cnn_pred_Y[k] == 0:
                    total_pred_Y[j] = 0
                
                # iterate through positive tree-predictions
                k += 1

        


        fig[fi] = plt.figure(f'Testing figure subject {s} on Model_{s}_{i}')
        fi += 1
        plt.suptitle(f'Subject <{s}>', fontsize = 24, y = 1)
        plt.title(f'Comparing ground-truth labels with predictions')
        plt.plot(total_pred_Y, color = 'orange')
        plt.plot(Test_Y, color = 'blue', alpha = 0.5)


        # Calculate evaluation coefficients spike-per-spike
        #cm_s_i = confusion_matrix(y_true = cnn_Y, y_pred = Pred_Y)
        cm_s_i = confusion_matrix(y_true = Test_Y, y_pred = total_pred_Y)
        cm_s.append(cm_s_i)

        #acc, prec, rec, f1s = calculate_metrics(cm = cm_s_i)
        #print_metrics(acc, prec, rec, f1s, 'Spike-per-spike')

    
    # Concatenate s-subjects confusion matrices with the rest
    cm_s_temp = np.round(np.mean(cm_s, axis = 0))
    cm_tot.append(cm_s_temp)

    acc, prec, rec, f1s = calculate_metrics(cm = cm_s_temp)
    print_metrics(acc, prec, rec, f1s, 'Spike-per-spike')

    # Save s-subject testing figures
    with open(fig_path + 'tstfig_' + str(s) + '.pkl', 'wb') as f:
        pkl.dump(fig, f)

print(f'\n> Total evaluation:\n')

# 1. Spike-per-spike method:
#cm_total = np.sum(cm_tot, axis = 0)
cm_total = np.round(np.sum(cm_tot, axis = 0))
Acc_tot, Prec_tot, Rec_tot, F1_tot = calculate_metrics(cm = cm_total)

print_metrics(Acc_tot, Prec_tot, Rec_tot, F1_tot, 'Spike-per-spike')
