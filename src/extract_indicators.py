from utils import *

with open(f'pickles{sls}testing_results.pkl', 'rb') as f:
	results = pkl.load(f)
	n_subjects = pkl.load(f)

true_spk_ind, true_spk_spd, pred_spk_ind, pred_spk_spd = spk_speed(results, n_subjects)


flag = False
while not flag:

	s = int(float(input(f'''
 > Enter subject\'s number to display its spikes\' speed.
   Available subject-numbers are [0-9].
   If you want to exit, enter any unavailable number.

 > Subject number: ''')))

	if s < 0 or s > 9:
		endreq = int(float(input(f'\n > You requested to exit. To proceed, enter 0: ')))
		if endreq == 0:
			flag = True
		else:
			clear()
			print(f' > Please, try again.')
	else:
		df = pd.DataFrame(true_spk_ind[s], columns = ['Index - GT'])
		df['Spike Speed - GT'] = true_spk_spd[s]
		df['Index - Pred'] = pred_spk_ind[s]
		df['Spike Speed - Pred'] = pred_spk_spd[s]
		print(df)
		ch = int(float(input(f'\n > To display more subjects, enter 1.\n > To exit, enter anything else.\n\n > Your choice: ')))
		if ch != 1:
			flag = True

print('\n > See ya!')
