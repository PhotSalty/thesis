from utils import *
from keras import backend as K
from keras.models import load_model
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

p = os.path.dirname(os.getcwd())
p1 = p + sls + 'data' + sls + 'pickle_output' + sls
datapkl = p1 + 'raw_data.pkl'

## Create testing figures path
epochs = 10
base_path = p + sls + 'Models' + sls + 'epochs_' + str(epochs) + sls + 'LOSO_10ep' + str(epochs) + sls
fig_path = base_path + 'Testing_figures' + sls
if not os.path.exists(fig_path):
	os.makedirs(fig_path)

with open(datapkl, 'rb') as f:
	windows = pkl.load(f)
	labels = pkl.load(f)
	tag = pkl.load(f)
	timestamps = pkl.load(f)
	means = pkl.load(f)
	stds = pkl.load(f)

## Subject list by their tags
subjects = np.unique(tag)


def test_subject(s):
	test_ind = np.asarray(np.where(tag == s))[0]

	Test_X = windows[test_ind[0]:test_ind[-1]+1, :, :]
	Test_Y = labels[test_ind[0]:test_ind[-1]+1]

	Test_X = apply_stadardization(windows = Test_X, means = means, stds = stds)
	print(f'\n################################ Test Data ready ################################################')

	ep = str(epochs)
	model_path = base_path + 'M' + s + '_epochs_' + ep + '.mdl'
	model = load_model(model_path)

	pred_Y = model.predict(x = Test_X)
	return pred_Y, Test_Y

def plot_pred(s, fig_path):
	## Comparing prediction with the signal-labels
	fig = plt.figure('Testing figure ' + s)
	plt.suptitle(f'Subject-{s}:',fontsize=24, y=1)
	plt.title(f'Comparing ground-truth labels with predictions',fontsize=16)
	plt.plot(pred_Y, color = 'orange')
	plt.plot(Test_Y, color = 'blue', alpha = 0.5)
	plt.plot(p, pred_Y[p, 0], 'x', color = 'green')

	fig.savefig(fig_path + 'Subj' + s + '.pdf')

def print_metrics(Acc, Prec, Rec, F1s, mtype):
	print(f'\n{mtype} metrics:') 
	print(f'''
	Accuracy  = {Acc:.3f}
	Precision = {Prec:.3f}
	Recall    = {Rec:.3f}
	F1-score  = {F1s:.3f}''')

## Threshold of true-prediction:	----------> Not picked yet
	# threshold = 0.8
	# pred_Y[np.where(pred_Y < threshold)] = 0
	# pred_Y[np.where(pred_Y >= threshold)] = 1

##   find_peaks() -> p    -> 1 positive prediction per predicted spike
## Windows_eval_coeffs(p) -> Spike-per-spike evaluation, eliminating
##                           the extra predictions of the same spike
## 
##   confusion_matrix()   -> Window-per-window evaluation, more accurate
##                           but not able to define "spike-area"
##

cm_sps = []
cm_wpw = []

for s in subjects:
	pred_Y, Test_Y = test_subject(s)
	# pred_Y[np.where(pred_Y >= 0.8)] = 1

## Random threshold pick = 0.8	
	pred_Y = np.where(pred_Y < 0.85, 0, 1)
	print(f'Session of {s} Subject:\n')

## Extract the peaks of the positive class:
	p, _ = find_peaks(pred_Y[:, 0], distance = 21)
	print(f'peaks: {p.shape}')
	print(f'pred_Y: {pred_Y.shape}')

## Plotting
	plot_pred(s, fig_path)

## Calculate evaluation coefficients spike-per-spike
	tp_sps, fp_sps, fn_sps, tn_sps = windows_eval_coeffs(testY = Test_Y, predY = pred_Y, pred_peaks = p)
	cms = np.array([[tn_sps, fp_sps], [fn_sps, tp_sps]])
	cm_sps.append(cms)

## Calculate evaluation coefficients window-per-window
	cmw = confusion_matrix(y_true = Test_Y, y_pred = pred_Y)
	tn_wpw, fp_wpw, fn_wpw, tp_wpw = cmw.ravel()
	cm_wpw.append(cmw)

	acc, prec, rec, f1s = calculate_metrics(cms)
	print_metrics(acc, prec, rec, f1s, 'Spike-per-spike')

	acc, prec, rec, f1s = calculate_metrics(cmw)
	print_metrics(acc, prec, rec, f1s, 'Window-per-window')



print(f'Confusion matrices of {subjects.shape[0]} subjects are calculated.')

# Spike-per-spike Aggregate cm:
cm_sps_sum = np.sum(cm_sps, axis = 0)
Acc_sps_sum, Prec_sps_sum, Rec_sps_sum, F1_sps_sum = calculate_metrics(cm = cm_sps_sum)

print_metrics(Acc_sps_sum, Prec_sps_sum, Rec_sps_sum, F1_sps_sum, 'Spike-per-spike Sum')

# Spike-per-spike Mean cm:
cm_sps_mean = np.mean(cm_sps, axis = 0)
Acc_sps_mean, Prec_sps_mean, Rec_sps_mean, F1_sps_mean = calculate_metrics(cm = cm_sps_mean)

print_metrics(Acc_sps_mean, Prec_sps_mean, Rec_sps_mean, F1_sps_mean, 'Spike-per-spike Mean')


# Window-per-window Aggregate cm:
cm_wpw_sum = np.sum(cm_wpw, axis = 0)
Acc_wpw_sum, Prec_wpw_sum, Rec_wpw_sum, F1_wpw_sum = calculate_metrics(cm = cm_wpw_sum)

print_metrics(Acc_wpw_sum, Prec_wpw_sum, Rec_wpw_sum, F1_wpw_sum, 'window-per-window Sum')

# Window-per-window Mean cm:
cm_wpw_mean = np.mean(cm_wpw, axis = 0)
Acc_wpw_mean, Prec_wpw_mean, Rec_wpw_mean, F1_wpw_mean = calculate_metrics(cm = cm_wpw_mean)

print_metrics(Acc_wpw_mean, Prec_wpw_mean, Rec_wpw_mean, F1_wpw_mean, 'window-per-window Mean')
