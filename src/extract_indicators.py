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
                tind = true_spk_ind[s]
                tspd = true_spk_spd[s]
                pind = pred_spk_ind[s]
                pspd = pred_spk_spd[s]
                print(np.shape(tind), np.shape(tspd))
                print(np.shape(pind), np.shape(pspd))
                while np.shape(tind) > np.shape(pind):
                    pind = np.append(pind, 0)
                    pspd = np.append(pspd, 0)
                while np.shape(pind) > np.shape(tind):
                    tind = np.append(tind, 0)
                    tspd = np.append(tspd, 0)
                
                df = pd.DataFrame(tind, columns = ['Index - GT'])
                df['Spike Speed - GT'] = tspd
                df['Index - Pred'] = pind
                df['Spike Speed - Pred'] = pspd
                print(df)

                ch = int(float(input(f'\n > To display more subjects, enter 1.\n > To exit, enter anything else.\n\n > Your choice: ')))
                if ch != 1:
                        flag = True

print('\n > See ya!')
