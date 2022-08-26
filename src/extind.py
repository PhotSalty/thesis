from utils import *

names = np.asarray(['sltn', 'gali', 'sdrf', 'pasx', 'anti', 'komi', 'fot', 'agge', 'conp', 'LH_galios'])

with open(f'pickles{sls}testing_results.pkl', 'rb') as f:
        results = pkl.load(f)
        n_subjects = pkl.load(f)

#true_spk_ind, true_spk_spd, pred_spk_ind, pred_spk_spd = spk_speed(results, n_subjects)

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
                name = np.asarray([names[s]])
                
                flag_ind = False
                while not flag_ind:
                        print(f'\n > You have selected <{name[0]}> subject.')
                        print(f'\n\t- for flight-time indicator, enter 0')
                        print(f'\n\t- for spike-power indicator, enter 1')

                        ind_type = int(float(input(f'\n\t> Your choice: ')))

                        if ind_type < 0 or ind_type > 1:
                                clear()
                                print(f' > Wrong indicator code.')
                        else:
                                flag_ind = True

                if ind_type == 1:

                        true_spk_ind, true_spk_spd, pred_spk_ind, pred_spk_spd = spk_speed_1(results[s], name)

                        tind = true_spk_ind
                        tspd = true_spk_spd
                        pind = pred_spk_ind
                        pspd = pred_spk_spd

                        #print(np.shape(tind), np.shape(tspd))
                        #print(np.shape(pind), np.shape(pspd))
                
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
                        plt.show()

                        ch = int(float(input(f'\n > To display more subjects, enter 1.\n > To exit, enter anything else.\n\n > Your choice: ')))
                        if ch != 1:
                                flag = True


                if ind_type == 0:
                        
                    print('\nNot ready.')

                    #tr_flt_time, tr_flt_ind, prd_flt_time, prd_flt_ind = flight_time(results[s], name)

                    #tflt = tr_flt_time
                    #tind = tr_flt_ind
                    #pflt = prd_flt_time
                    #pind = prd_flt_ind


print('\n > See ya!')
