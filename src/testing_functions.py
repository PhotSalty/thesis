
from utils import *

from keras import backend as K
from keras.models import load_model
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


### Display given method's metrics
def print_metrics(Acc, Prec, Rec, F1s, mtype):
	print(f'\n > {mtype} metrics:') 
	print(f'''
\tAccuracy  = {Acc:.3f}
\tPrecision = {Prec:.3f}
\tRecall    = {Rec:.3f}
\tF1-score  = {F1s:.3f}\n'''.expandtabs(6))


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
		model_path = mdl_path + 'M' + st + '_' + n_subjects + '.mdl'

	# Load model and perform testing
		model = load_model(model_path)
		pred_Y = model.predict(x = Test_X)

	# Complete prediction with pre-calculated threshold -> 0.6
		Pred_Y = np.where(pred_Y < 0.6, 0, 1)

	# Find positive-class peaks, in order to eliminate multiple "positive" windows for one spike
		p, _ = find_peaks(Pred_Y[:, 0], distance = 20)
		pos_pred = np.where(Pred_Y == 1)
		print(f'\n\tCalculated {p.shape} Peaks, out of {pos_pred.shape} positive windows'.expandtabs(4))

	# Construct and save a plot for the prediction results
		fig = plt.figure('Testing figure ' + st)
		plt.suptitle(f'Subject <{st}>', fontsize = 24, y = 1)
		plt.title(f'Comparing ground-truth labels with predictions')
		plt.plot(Pred_Y, color = 'orange')
		plt.plot(Test_Y, color = 'blue', alpha = 0.5)
		plt.plot(p, Pred_Y[p, 0], 'x', color = 'green', linewidth = 1.5)

		fig.savefig(fig_path + 'Subj' + st + '.pdf')

	# Calculate evaluation coefficients spike-per-spike
		tp_sps, fp_sps, fn_sps, tn_sps = windows_eval_coeffs(testY = Test_Y, predY = pred_Y, pred_peaks = p)
		cms = np.array([[tn_sps, fp_sps], [fn_sps, tp_sps]])
		cm_sps.append(cms)

	# Calculate evaluation coefficients window-pre-window
		cmw = confusion_matrix(y_true = Test_Y, y_pred = pred_Y)
		# tn_wpw, fp_wpw, fn_wpw, tp_wpw = cmw.ravel()
		cm_wpw.append(cmw)

		acc, prec, rec, f1s = calculate_metrics(cms)
		print_metrics(acc, prec, rec, f1s, 'Spike-per-spike')

		acc, prec, rec, f1s = calculate_metrics(cmw)
		print_metrics(acc, prec, rec, f1s, 'Window-per-window')

	
	print(f'\n Confusion matrices of {subject_tags.shape[0]} subjects')

# Total Confusion matrices and evaluation:
	
	# 1. Spike-per-spike method:
	cm_sps_tot = np.sum(cm_sps, axis = 0)
	Acc_sps_tot, Prec_sps_tot, Rec_sps_tot, F1_sps_tot = calculate_metrics(cm = cm_sps_tot)
	
	print_metrics(Acc_sps_tot, Prec_sps_tot, Rec_sps_tot, F1_sps_tot, 'Spike-per-spike')
	
	# 2. Winodw-per-window method:
	cm_wpw_tot = np.sum(cm_wpw, axis = 0)
	Acc_wpw_tot, Prec_wpw_tot, Rec_wpw_tot, F1_wpw_tot = calculate_metrics(cm = cm_wpw_tot)

	print_metrics(Acc_wpw_tot, Prec_wpw_tot, Rec_wpw_tot, F1_wpw_tot, 'Window-per-window')