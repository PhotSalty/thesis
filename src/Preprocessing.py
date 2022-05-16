from re import X
from sys import getsizeof
from utils import *
from filterfuncs import *

names = ['sltn', 'sdrf', 'gali', 'pasx', 'anti', 'komi', 'fot']
# names = ['sltn', 'sdrf']

# Print full-base statistics:
base_stats(names)

# Concatenate subjects:
subjects, ns, po, ne = subj_tot(names)
ns = np.array(ns) - 1

# Balance data - Positive-class frames augmentation:
newsubj, ind = balance_windows(subjects, ns, po, ne)

full = np.array([], dtype = object)


wds = deepcopy(newsubj[0].windows)
lbl = deepcopy(newsubj[0].labels)
tg = np.full(np.shape(lbl), newsubj[0].tag)
tmst = deepcopy(newsubj[0].timestamps)

for i in np.arange(len(newsubj))[1:]:
	wds = np.append(wds, newsubj[i].windows, axis = 0)
	lbl = np.append(lbl, newsubj[i].labels, axis = 0)
	temp = np.full(np.shape(lbl), newsubj[i].tag)
	tg = np.append(tg, temp, axis = 0)
	tmst = np.append(tmst, newsubj[i].timestamps, axis = 0)

full = np.array([wds, lbl, tg, tmst], dtype = object)
# full = np.array([newsubj[0].windows, newsubj[0].labels, newsubj[0].tag, newsubj[0].timestamps], dtype=object)

p = os.path.dirname(os.getcwd())
p1 = p + '\\data\\pickle_output\\'
datap = p1 + 'full_data' + '.pkl'
with open(datap, 'wb') as f:
	pkl.dump(full, f)

# with open("data_saltidis_n.pkl", 'rb') as f:
# 	data = pkl.load(f)