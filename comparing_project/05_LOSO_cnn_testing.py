
from utils import *

e_impact = 0.000125

with open('tree_out_data.pkl', 'rb') as f:
    a = pkl.load(f)
    b = pkl.load(f)
    c = pkl.load(f)
    tst_orig = pkl.load(f)
    tst_auix = pkl.load(f)
    tst_lbls = pkl.load(f)

for s in np.arange(9):

    Test_orig = tst_orig[s]
    Test_auxi = tst_auxi[s]
    Test_lbls = tst_lbls[s]

    Test_X, Test_Y, Test_ind = extract_indicators(Test_orig, Test_auxi, Test_lbls, e_impact)

    
