from utils_case import *

from metacost import MetaCost
from imblearn.over_sampling import SMOTE
from collections import Counter


# 1. Load preprocessed data
pkl_path = 'preprocessed_data' + sls + 'prepdata.pkl'
with open(pkl_path, 'rb') as f:
	s_orig = pkl.load(f)
	s_auxi = pkl.load(f)
	labels = pkl.load(f)
	e_impact = pkl.load(f)
	swing_interval = pkl.load(f)


tst_subj = 0
trn9_subj = np.arange(len(s_orig)) != tst_subj
# print(trn9_subj)


###### 9 subjects for LOSO

tst1_orig = s_orig[tst_subj, 0]
tst1_auxi = s_auxi[tst_subj, 0]
tst1_lbls = labels[tst_subj]

trn9_orig = s_orig[trn9_subj, 0]
trn9_auxi = s_auxi[trn9_subj, 0]
trn9_lbls = labels[trn9_subj]

def plot_trtst_signals(i, j):
	fig = plt.figure('Testing Subject')
	plt.plot(tst1_orig, 'darkblue')
	plt.plot(tst1_lbls*14, 'orange', linewidth = 1.5)

	fig = plt.figure('Testing Subject 9')
	plt.plot(trn9_orig[i], 'darkblue')
	plt.plot(trn9_lbls[i]*14, 'orange', linewidth = 1.5)

	fig = plt.figure('Training Subject 10')
	plt.plot(trn9_orig[j], 'darkblue')
	plt.plot(trn9_lbls[j]*14, 'orange', linewidth = 1.5)

	plt.show()

# plot_trtst_signals(7, 8)


###### 8 subjects for tree

val_subj = 0	#1
trn8_subj = np.arange(len(trn9_orig)) != val_subj

val_orig = trn9_orig[val_subj]
val_auxi = trn9_auxi[val_subj]
val_lbls = trn9_lbls[val_subj]

trn8_orig = trn9_orig[trn8_subj]
trn8_auxi = trn9_auxi[trn8_subj]
trn8_lbls = trn9_lbls[trn8_subj]

print(trn8_orig.shape, trn8_orig[0].shape)

trn_full_orig = concatenate_lists(trn8_orig, mde = 1)
trn_full_auxi = concatenate_lists(trn8_auxi, mde = 0)
trn_full_lbls = concatenate_labels(trn8_lbls)

print(trn_full_orig.shape, trn_full_auxi.shape, trn_full_lbls.shape)

Tree8_X, Tree8_Y, Tree8_ind = extract_indicators(trn_full_orig, trn_full_auxi, trn_full_lbls, e_impact)

print(Tree8_X.shape, Tree8_Y.shape, Tree8_ind.shape)

def plot_tree_signal():
	plt.plot(trn_full_orig[:220001], color = 'darkblue')
	for i in Tree8_ind[:600]:
		plt.axvspan(xmin = i-2*64, xmax = i+2*64, color = 'green', alpha = 0.2)
	plt.show()

# plot_tree_signal()

alg = tree.DecisionTreeClassifier()
C = np.array([[0, 1000], [0, 0]])
S = pd.DataFrame(Tree8_X, columns = ['indi1', 'indi2', 'indi3', 'indi4', 'indi5', 'indi6'])
S['Target'] = Tree8_Y
dtree_model = MetaCost(S, alg, C).fit('Target', 3)


Val8_X, Val8_Y, Val8_ind = extract_indicators(val_orig, val_auxi, val_lbls, e_impact)
# Test decision tree with the Test1 (1 subject out of 9 -> Dtree LOSO)
pred = dtree_model.predict(Val8_X)

# plt.plot(Val8_Y, color = 'darkblue')
# plt.plot(pred, color = 'orange', alpha = 0.5)
# plt.show()

pred_pos = Val8_ind[np.nonzero(pred)]
print(pred_pos.shape)

wds, lbls, w_ind = extract_event_windows(pred_pos, val_orig, val_lbls)

print(wds.shape, lbls.shape, w_ind.shape)

print(np.shape(np.nonzero(lbls)))

spk_windows = wds[np.nonzero(lbls)]
print(spk_windows.shape)

fig, axs = plt.subplots(2, 3)
axs[0, 0].plot(spk_windows[3])
axs[0, 1].plot(spk_windows[16])
axs[0, 2].plot(spk_windows[22])
axs[1, 0].plot(spk_windows[38])
axs[1, 1].plot(spk_windows[45])
axs[1, 2].plot(spk_windows[57])
plt.show()