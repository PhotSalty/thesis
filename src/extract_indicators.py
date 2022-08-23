from utils import *

with open(f'pickles{sls}testing_results.pkl', 'rb') as f:
	results = pkl.load(f)
	n_subjects = pkl.load(f)

true_spk_ind, true_spk_spd, pred_spk_ind, pred_spk_spd = spk_speed(results, n_subjects)

