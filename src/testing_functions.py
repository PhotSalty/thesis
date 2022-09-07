
from utils import *

from keras import backend as K
from keras.models import load_model
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score
from prep_windows import prepare_oversampled_windows, prepare_widesampled_windows

### Display given method's metrics
def print_metrics(Acc, Prec, Rec, F1s, mtype):
        print(f'\n > {mtype} metrics:') 
        print(f'''
\tWeighted Accuracy   ->   {Acc:.3f}
\tPrecision           ->   {Prec:.3f}
\tRecall              ->   {Rec:.3f}
\tF1-score            ->   {F1s:.3f}\n'''.expandtabs(6))


### One subject testing:
def solo_test(Test_X, means, stds, mdl_path, tg): #, Test_Y, fig_path, tg):

        if tg < 10:
                tg = '0' + str(tg)
        else:
                tg = str(tg)

        print(f'\n > Testing subject <{tg}>:')

        # fig_path += 's{tg}_testing' + sls
        # if not os.path.exists(fig_path):
        #         os.makedirs(fig_path)

        Test_X = apply_standardization(Test_X, means, stds)

        model_path = mdl_path + 'M' + tg + '_10' + '.mdl'

        model = load_model(model_path)
        pred_Y = model.predict(x = Test_X)

        return pred_Y



def LOSO_testing1(windows, labels, tgz, means, stds, epochs, n_subjects, mdl_path):

        print(f'\n>> Testing')

# List of available subject tags
        names = np.array(['sltn', 'gali', 'sdrf', 'pasx', 'anti', 'komi', 'fot', 'agge', 'conp', 'LH_galios'])
        tags = np.array(['01', '02', '03', '04', '05', '06', '07', '08', '09', '10'])

        fig_path = mdl_path + 'Testing_figures' + sls

        if not os.path.exists(fig_path):
                os.makedirs(fig_path)

# Define confusion matrices for spike-per-spike and window-per window methods
        cm_sps = []
        cm_wpw = []

# figure list for pickle        
        fig = np.zeros(10, dtype = object)
        fi = 0

# # Balanced accuracy total weight
#       is1_tot = 0
#       is0_tot = 0
        
        sampling_method = int(float(input(f'\n > Select windows extracting method:\n\n\t1. super-sampling\n\n\t2. wide-sampling\n\n\t3. biased-sampling\n\n\t> Your choice: '.expandtabs(6))))
# Saving results:
        results = np.zeros((n_subjects, 3), dtype = object)
        ri = 0

# LOSO iterations through subjects
        ovl = float(input(f'\n\t Select overlap : ').expandtabs(6))
        if ovl > 1:
            ovl = ovl / 100
        if ovl < 0:
            ovl = -ovl
        if ovl == 0:
            print('\n\tOverlap auto mode -> 0.4.'.expandtabs(6))
            ovl = 0.4

        for st, nm in zip(tags, names):

                print(f'\n > Session {st}: leaving subject <{st}> out')

        # Define testing subject's indices
                
                if sampling_method == 1:
                    subject = prepare_oversampled_windows(nm, st)
                    Test_X = subject.windows
                    Test_Y = subject.labels
                elif sampling_method == 2:
                    subject = prepare_widesampled_windows(nm, st, ovl)
                    Test_X = subject.windows
                    Test_Y = subject.labels
                else:
                    ind = np.asarray(np.where(tgz == st))[0]
                    Test_X = windows[ind, :, :]
                    Test_Y = labels[ind]


                Test_X = apply_standardization(windows = Test_X, means = means, stds = stds)

        # Define model path
                model_path = mdl_path + 'M' + st + '_' + str(n_subjects) + '.mdl'

        # Load model and perform testing
                model = load_model(model_path)
                pred_Y = model.predict(x = Test_X)

        # Complete prediction with pre-calculated threshold -> 0.6
                Pred_Y = np.where(pred_Y < 0.6, 0, 1)

        # Find positive-class peaks, in order to eliminate multiple "positive" windows for one spike
                p, _ = find_peaks(Pred_Y[:, 0], distance = 22)
                pos_pred = np.asarray(np.where(Pred_Y == 1))
                print(f'\n\tCalculated {p.shape} Peaks, out of {pos_pred.shape} positive windows'.expandtabs(4))

                results[ri, 0] = Test_Y
                results[ri, 1] = pred_Y
                results[ri, 2] = Test_X
                ri += 1                

        # Construct and save a plot for the prediction results
                fig[fi] = plt.figure('Testing figure ' + st)
                fi += 1
                plt.suptitle(f'Subject <{st}>', fontsize = 24, y = 1)
                plt.title(f'Comparing ground-truth labels with predictions')
                plt.plot(Pred_Y, color = 'orange')
                plt.plot(Test_Y, color = 'blue', alpha = 0.5)
                plt.plot(p, Pred_Y[p, 0], 'x', color = 'green', linewidth = 1.5)
                # fig.savefig(fig_path + 'Subj' + st + '.pdf')

        # Calculate evaluation coefficients spike-per-spike
                tp_sps, fp_sps, fn_sps, tn_sps = windows_eval_coeffs(testY = Test_Y, predY = Pred_Y, pred_peaks = p)
                cms = np.array([[tn_sps, fp_sps], [fn_sps, tp_sps]])
                cm_sps.append(cms)

        # Calculate evaluation coefficients window-pre-window
                cmw = confusion_matrix(y_true = Test_Y, y_pred = Pred_Y)
                # tn_wpw, fp_wpw, fn_wpw, tp_wpw = cmw.ravel()
                cm_wpw.append(cmw)

        # Balanced accuracy weight
                # is1 = np.count_nonzero(Test_Y)
                # is0 = np.shape(Test_Y)[0] - is1
                # weight = is0 / is1

                acc, prec, rec, f1s = calculate_metrics(cm = cms)
                print_metrics(acc, prec, rec, f1s, 'Spike-per-spike')

                acc, prec, rec, f1s = calculate_metrics(cm = cmw)
                print_metrics(acc, prec, rec, f1s, 'Window-per-window')
                
                # wacc = balanced_accuracy_score(Test_Y, Pred_Y)
                # print(f'\nWeighted accuracy = {wacc}\n')

                # is1_tot += is1
                # is0_tot += is0
        
        with open(fig_path + 'tstfig1.pkl', 'wb') as f:
                pkl.dump(fig, f)
                
        print(f'\n Confusion matrices of {tags.shape[0]} subjects')
        
        # Close figures - prevent from showing
        plt.close('all')


        # # Total weight
        # weight_tot = is0_tot / is1_tot

# Total Confusion matrices and evaluation:
        
        # 1. Spike-per-spike method:
        cm_sps_tot = np.sum(cm_sps, axis = 0)
        Acc_sps_tot, Prec_sps_tot, Rec_sps_tot, F1_sps_tot = calculate_metrics(cm = cm_sps_tot)
        
        print_metrics(Acc_sps_tot, Prec_sps_tot, Rec_sps_tot, F1_sps_tot, 'Spike-per-spike')
        
        # 2. Winodw-per-window method:
        cm_wpw_tot = np.sum(cm_wpw, axis = 0)
        Acc_wpw_tot, Prec_wpw_tot, Rec_wpw_tot, F1_wpw_tot = calculate_metrics(cm = cm_wpw_tot)

        print_metrics(Acc_wpw_tot, Prec_wpw_tot, Rec_wpw_tot, F1_wpw_tot, 'Window-per-window')

        return results


### Testing subjects' models and display total results
def LOSO_testing(windows, labels, tag, means, stds, epochs, n_subjects, mdl_path):

        print(f'\n>> Testing')

# List of available subject tags
        subject_tags = np.unique(tag)

        fig_path = mdl_path + 'Testing_figures' + sls

        if not os.path.exists(fig_path):
                os.makedirs(fig_path)

# Define confusion matrices for spike-per-spike and window-per window methods
        cm_sps = []
        cm_wpw = []

# figure list for pickle        
        fig = np.zeros(10, dtype = object)
        fi = 0

# # Balanced accuracy total weight
#       is1_tot = 0
#       is0_tot = 0

# Saving results:
        results = np.zeros((n_subjects, 3), dtype = object)
        ri = 0

# LOSO iterations through subjects
        for st in subject_tags:

                print(f'\n > Session {st}: leaving subject <{st}> out')

        # Define testing subject's indices
                ind_tst = np.asarray(np.where(tag == st))[0]

        # Define testing windows and their labels 
                Test_X = windows[ind_tst, :, :]
                Test_Y = labels[ind_tst]

                Test_X = apply_standardization(windows = Test_X, means = means, stds = stds)

        # Define model path
                model_path = mdl_path + 'M' + st + '_' + str(n_subjects) + '.mdl'

        # Load model and perform testing
                model = load_model(model_path)
                pred_Y = model.predict(x = Test_X)

        # Complete prediction with pre-calculated threshold -> 0.6
                Pred_Y = np.where(pred_Y < 0.6, 0, 1)

        # Find positive-class peaks, in order to eliminate multiple "positive" windows for one spike
                p, _ = find_peaks(Pred_Y[:, 0], distance = 22)
                pos_pred = np.asarray(np.where(Pred_Y == 1))
                print(f'\n\tCalculated {p.shape} Peaks, out of {pos_pred.shape} positive windows'.expandtabs(4))

                results[ri, 0] = Test_Y
                results[ri, 1] = pred_Y
                results[ri, 2] = Test_X
                ri += 1                

        # Construct and save a plot for the prediction results
                fig[fi] = plt.figure('Testing figure ' + st)
                fi += 1
                plt.suptitle(f'Subject <{st}>', fontsize = 24, y = 1)
                plt.title(f'Comparing ground-truth labels with predictions')
                plt.plot(Pred_Y, color = 'orange')
                plt.plot(Test_Y, color = 'blue', alpha = 0.5)
                plt.plot(p, Pred_Y[p, 0], 'x', color = 'green', linewidth = 1.5)
                # fig.savefig(fig_path + 'Subj' + st + '.pdf')

        # Calculate evaluation coefficients spike-per-spike
                tp_sps, fp_sps, fn_sps, tn_sps = windows_eval_coeffs(testY = Test_Y, predY = Pred_Y, pred_peaks = p)
                cms = np.array([[tn_sps, fp_sps], [fn_sps, tp_sps]])
                cm_sps.append(cms)

        # Calculate evaluation coefficients window-pre-window
                cmw = confusion_matrix(y_true = Test_Y, y_pred = Pred_Y)
                # tn_wpw, fp_wpw, fn_wpw, tp_wpw = cmw.ravel()
                cm_wpw.append(cmw)

        # Balanced accuracy weight
                # is1 = np.count_nonzero(Test_Y)
                # is0 = np.shape(Test_Y)[0] - is1
                # weight = is0 / is1

                acc, prec, rec, f1s = calculate_metrics(cm = cms)
                print_metrics(acc, prec, rec, f1s, 'Spike-per-spike')

                acc, prec, rec, f1s = calculate_metrics(cm = cmw)
                print_metrics(acc, prec, rec, f1s, 'Window-per-window')
                
                # wacc = balanced_accuracy_score(Test_Y, Pred_Y)
                # print(f'\nWeighted accuracy = {wacc}\n')

                # is1_tot += is1
                # is0_tot += is0



        
        with open(fig_path + 'tstfig.pkl', 'wb') as f:
                pkl.dump(fig, f)
                
        print(f'\n Confusion matrices of {subject_tags.shape[0]} subjects')
        
        # Close figures - prevent from showing
        plt.close('all')


        # # Total weight
        # weight_tot = is0_tot / is1_tot

# Total Confusion matrices and evaluation:
        
        # 1. Spike-per-spike method:
        cm_sps_tot = np.sum(cm_sps, axis = 0)
        Acc_sps_tot, Prec_sps_tot, Rec_sps_tot, F1_sps_tot = calculate_metrics(cm = cm_sps_tot)
        
        print_metrics(Acc_sps_tot, Prec_sps_tot, Rec_sps_tot, F1_sps_tot, 'Spike-per-spike')
        
        # 2. Winodw-per-window method:
        cm_wpw_tot = np.sum(cm_wpw, axis = 0)
        Acc_wpw_tot, Prec_wpw_tot, Rec_wpw_tot, F1_wpw_tot = calculate_metrics(cm = cm_wpw_tot)

        print_metrics(Acc_wpw_tot, Prec_wpw_tot, Rec_wpw_tot, F1_wpw_tot, 'Window-per-window')

        return results
