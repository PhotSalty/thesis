# if int(sys.argv[1]) == 1:
# 	import matplotlib
# 	matplotlib.use('TkAgg')
# 	from matplotlib import pyplot as plt
# 	sls = '/'
# 	print('We are in Ubuntu!')
# else:
# 	import matplotlib.pyplot as plt
# 	sls = '\\'

# from sys import getsizeof
# from filterfuncs import *
# import sys
# sls = sys.argv[1]

from filterfuncs import base_stats
from utils import *

names = ['sltn', 'sdrf', 'gali', 'pasx', 'anti', 'komi', 'fot']
# names = ['sltn', 'sdrf']

# Print full-base statistics:
base_stats(names)

# Concatenate subjects:
subjects, ns, po, ne = subj_tot(names)
ns = np.array(ns) - 1

# Balance data - Positive-class frames augmentation:
newsubj, ind = balance_windows(subjects, ns, po, ne)

wds = deepcopy(newsubj[0].windows)
lbl = deepcopy(newsubj[0].labels)
tg = np.full(np.shape(newsubj[0].labels), newsubj[0].tag)
tmst = deepcopy(newsubj[0].timestamps)
# print("1.", np.shape(newsubj[0].windows), np.shape(newsubj[0].labels), np.shape(tg), np.shape(newsubj[0].timestamps))

for i in np.arange(len(newsubj))[1:]:
	wds = np.append(wds, newsubj[i].windows, axis = 0)
	lbl = np.append(lbl, newsubj[i].labels, axis = 0)
	temp = np.full(np.shape(newsubj[i].labels), newsubj[i].tag)
	tg = np.append(tg, temp, axis = 0)
	tmst = np.append(tmst, newsubj[i].timestamps, axis = 0)
	# print(f'{i+1}. {np.shape(newsubj[i].windows)}, {np.shape(newsubj[i].labels)}, {np.shape(temp)}, {np.shape(newsubj[i].timestamps)}')

	means, stds = standardization_parameters(wds)

# After fixing tmst size, I get an error creating an object with wds + the other
# full = np.array([wds, lbl, tg, tmst], dtype = object)


p = os.path.dirname(os.getcwd())

p1 = p + sls + 'data' + sls + 'pickle_output' + sls
datap = p1 + 'full_data' + '.pkl'
# datap = 'full_data.pkl'
with open(datap, 'wb') as f:
	pkl.dump(wds, f)
	pkl.dump(lbl, f)
	pkl.dump(tg, f)
	pkl.dump(tmst, f)
	pkl.dump(means, f)
	pkl.dump(stds, f)

# with open(datap, 'rb') as f:
# 	wds1 = pkl.load(f)
# 	lbl1 = pkl.load(f)
# 	tg1 = pkl.load(f)
# 	tmst1 = pkl.load(f)

# def comp(A, B, s):
# 	if np.array_equal(A,B):
# 		print(f'{s} Yey!')
# 	else:
# 		print(f'{s} Ney!')

# ss = ['Windows', 'Labels', 'Tags', 'Timestamps']
# comp(wds1, wds, ss[0])
# comp(lbl1, lbl, ss[1])
# comp(tg1, tg, ss[2])
# comp(tmst1, tmst, ss[3])




