#### PRE-RUN DEFINITIONS #######################################################################################


# 0. Libraries used
import random as rand
from math import floor, ceil
import os
from turtle import speed
import numpy as np
import pandas as pd
import pickle as pkl
from scipy.signal import filtfilt, butter, find_peaks, argrelextrema
from copy import deepcopy
import sys
import platform
from sklearn.metrics import confusion_matrix


# 1. Operating System check
psys = platform.system()
print(f'\n> Your Operating system is: {psys}')
if psys == 'Linux':
        import matplotlib
        matplotlib.use('TkAgg')
        from matplotlib import pyplot as plt
        sls = '/'
        print(f'\n\tLinux path-type: slash  =   \'{sls}\''.expandtabs(6))
        clear = lambda: os.system('clear')
        print(f'\n\tLinux clear cmd command = \'clear\''.expandtabs(6))
elif psys == 'Windows':
        import matplotlib.pyplot as plt
        sls = '\\'
        print(f'\n\tWindows path-type: slash  =  \'{sls}\''.expandtabs(6))
        clear = lambda: os.system('cls')
        print(f'\n\tWindows clear cmd command = \'cls\''.expandtabs(6))
else:
        sys.exit(f'\n > Not supported Operating System [Linux, Windows]\n > Please, check utils.py')



# 2. Spike-annotation method
def case_selection():
        print(f'''\n
> Select spike annotation method:

\t1. WITH slow-shots and WITH jump-services

\t2. WITH slow-shots but WITHOUT jump_services

\t3. WITHOUT slow-shots but WITH jump-services

\t4. WITHOUT slow-shots and WITHOUT jump-services\n'''.expandtabs(6))
        
        case = int(input(f'  Please, type only the case number (1-4): '))
        print(' ')

        return case

flag = False
while not flag:
        case_number = case_selection()
        if case_number == 1 or case_number == 2 or case_number == 3 or case_number == 4:
                flag = True
        else:
                clear()
                print(f'\n  Wrong input, please try again.')

if case_number == 1:
        file_folder = 'ss_js'
elif case_number == 2:
        file_folder = 'ss_no-js'
elif case_number == 3:
        file_folder = 'no-ss_js'
elif case_number == 4:
        file_folder = 'no-ss_no-js'


################################################################################################################








#### A CLASS TO COLLECT AND EDDIT RECORDING DATA ################################################################


class recording:

# 0. Initialization and basic methods
        def __init__(self, name, tag):
                self.name = name
                self.tag = tag
                self.fs = 64
                self.raw_acc = None
                self.raw_ang = None
                self.t = None
                self.spikes = None
                self.filt_acc = None
                self.filt_ang = None
                if name.find('LH') == -1:
                        self.left_hand = False  #default
                else:
                        self.left_hand = True

                self.windows = None
                self.labels = None
                self.timestamps = None

        def set_signals(self, racc, rang, t, spks, facc, fang):
                self.raw_acc = racc
                self.raw_ang = rang
                self.t = t
                self.spikes = spks
                self.filt_acc = facc
                self.filt_ang = fang

        def set_windows(self, wds, lbls, tmst):
                self.windows = wds
                self.labels = lbls
                self.timestamps = tmst

        def copy(self):
                q = recording(self.name, self.tag)
                q.set_signals(self.raw_acc, self.raw_ang, self.t, self.spikes, self.filt_acc, self.filt_ang)
                q.set_windows(self.windows, self.labels, self.timestamps)
                return q

# 1. Get data
        def read_data(self):
                p = os.path.dirname(os.getcwd())
                p1 = p + sls + 'data' + sls + 'pickled_data' + sls + file_folder + sls
                p2 = p + sls + 'data' + sls + 'recordings' + sls
                datap = p1 + self.name + '.pkl'
                
                with open(datap, 'rb') as f:
                        data = pkl.load(f)

                spikes = data[0]
                # blocks = data[1]
                # services = data[2]

                datar = p2 + self.name
                data1 = np.fromfile(datar+"acc.bin", dtype=np.int16, sep='')
                data2 = np.fromfile(datar+"angularrate.bin", dtype=np.int16, sep='')

                acc = data1.reshape(len(data1)//3, 3)
                ang = data2.reshape(len(data2)//3, 3)

                acc = acc * 0.0004
                ang = ang * 0.07

                spks = spikes[:, [1,2]]
                # blcks = blocks
                # srvs = services
                t = np.arange(len(acc)) / self.fs

                spks = np.array(spks*self.fs)
                spks[:, 0] = np.round(spks[:, 0])
                spks[:, 1] = np.ceil(spks[:, 1])
                
                # Hand mirroring for left-handed subjects. Converting their 
                # spike signals to an identical right-handed subject's signal
                if self.left_hand:
                        acc, ang = hand_mirroring_signals(acc, ang)

                self.t = t
                self.raw_acc = acc
                self.raw_ang = ang
                self.spikes = spks.astype(int)

# 2. Filtering methods
        def mav_filter(self, signal, c):
                ret = signal.copy()
                w = int(np.ceil(64 * c))
                for i in np.arange(3):
                        ret[:, i] = np.convolve(signal[:, i], np.ones(w)/w, 'same')
                
                return ret
        
        def hp_filter(self, signal, fs):
                # 1st method - Butter parameters:
                nyq = fs / 2
                cutoff = 1 / nyq
                b, a = butter(6, cutoff, btype='high', analog=False)
                y1 = signal.copy()

                for i in np.arange(3):
                        y1[:, i] = filtfilt(b, a, signal[:, i])
                
                return y1

        def filtering(self):
                # (1) Moving average filter:
                acc_smth = self.mav_filter(self.raw_acc, c=.25)
                #ang_smth = self.mav_filter(self.raw_ang, c=.25)
                # (2) High-pass filter:
                facc = self.hp_filter(acc_smth, self.fs)
                #fang = self.hp_filter(ang_smth, self.fs)
                #facc = self.hp_filter(self.raw_acc, self.fs)
                #fang = self.hp_filter(self.raw_ang, self.fs)

                self.filt_acc = facc
                #self.filt_ang = fang
                self.filt_ang = self.raw_ang

# 3. Windowing methods
        def find_step_and_label(self, spks_sample, j, wd_smpl_len, s):
                ls = s + wd_smpl_len
                flag = True
                e = round(0.2 * 64)
                ovl = 0.6
                while flag:
                        bound = ls - spks_sample[j, 1]
                        if bound < 0:
                                if -bound <= wd_smpl_len: # ls > floor(spks_sample[j, 0]):
                                        step = -bound
                                        # step = np.abs(spks_sample[j, 0] - s) - 1
                                else:
                                        step = round(wd_smpl_len*ovl)
                                        # step = floor(wd_smpl_len*0.4)
                                flag = False
                                label = 0
                        elif bound <= e: # bound >= 0 and bound <= e:
                                step = 1
                                flag = False
                                label = 1
                        else:
                                if j < len(spks_sample[:, 0])-1:
                                        j+=1
                                        # print(j)
                                else:
                                        step = round(wd_smpl_len*ovl)
                                        # step = floor(wd_smpl_len*0.4)
                                        flag = False
                                        label = 0

                return step, label

        def windowing(self, acc, ang, spks, wd_smpl_len):
                s_tot = ceil(len(acc[:, 0]))
                ## The maximum length of windows = total samples
                ## Each window will contain wd_smpl_len samples
                wds = np.zeros([s_tot, wd_smpl_len, 6])
                # wds = np.zeros([2*s_tot, wd_smpl_len, 6])
                labels = deepcopy(wds[:, 0, 0])
                timestamps = deepcopy(labels)
                acc_temp = np.vstack( ( acc, np.zeros([62, 3]) ) )
                ang_temp = np.vstack( ( ang, np.zeros([62, 3]) ) )
                i = 0
                j = 0
                s = 0

                while s <= s_tot:
                        ls = s + wd_smpl_len
                        # if i >= 2 * s_tot:
                        #         print(i)
                        #         print(np.nonzero(labels)[0])
                        wds[i, :, 0:3] = deepcopy(acc_temp[s:ls, :])
                        wds[i, :, 3:] = deepcopy(ang_temp[s:ls, :])
                        timestamps[i] = s / 64
                        step, labels[i] = self.find_step_and_label(spks, j, wd_smpl_len, s)
                        
                        i += 1
                        s = s + step

                ## Finding the unused slots in the rear of wds
                zer = np.zeros([wd_smpl_len, 6], dtype = type(wds[0, :, :]))
                c = 1              
                while np.array_equal(wds[-c, :, :], zer):
                        c += 1

                ## We exclude the last window, losing a constructed non-spike
                ## window (mixed signal-tuples with zero-tuples)
                wds = wds[1:-c, :, :]
                labels = labels[1:-c]
                timestamps = timestamps[1:-c]
                self.windows = deepcopy(wds)

                # fig, axs = plt.subplots(2)
                # axs[0].plot(acc[-44:, :])
                # axs[1].plot(self.windows[-1, :, 0:3])
                # plt.show()

                self.labels = deepcopy(labels)
                self.timestamps = deepcopy(timestamps)

        def extend_windows(self, extra, n):
                self.windows = np.append(self.windows, extra, axis=0)
                self.labels  = np.append(self.labels, np.ones(n, dtype=np.int16))
                self.timestamps = np.append(self.timestamps, np.zeros(n, dtype=np.float64))


################################################################################################################








#### PREPROCESSING #############################################################################################

# 1. Select window length out of the given subjects
def calc_window_length(names):
        
        sp_len_tot = []
        
        for i, name in zip(np.arange(len(names)), names):

                if i < 9:
                        tag = '0' + str(i+1)
                else:
                        tag = str(i+1)
                
                subject = recording(name, tag)
                subject.read_data()

                sp_len = subject.spikes[:, 1] - subject.spikes[:, 0]
                
                sp_len_tot.append(sp_len)

        window_length = round(np.mean(sp_len_tot))

        return window_length


# 2. Initialize all subjects' attributes and collect them in "subjects" object
def subjects_init(names):
        
        print(f'\n# Subjects Before augmentation:')

        subjects = list()
        ns = list()
        po = list()
        ne = list()
        subjects_spike_length = list()
        pos = 0
        neg = 0

        for i in np.arange(len(names)):
                
                n = names[i]

                if i < 9:
                        tg = '0' + str(i+1)
                else:
                        tg = str(i+1)

                subjects.append(recording(name = n, tag = tg))

                subjects[i].read_data()
                subjects[i].filtering()

                # Calculate average spike-length for each subject
                spikes_length = subjects[i].spikes[:,1] - subjects[i].spikes[:,0]
                # spikes_length = spikes_length * subjects[i].fs				# multiplied with freq in order to convert time to samples
                # subjects_spike_length.append(np.mean(spikes_length))
                subjects_spike_length.extend(spikes_length)
        
        # Extract mean/median spike-length of all subjects
        # window_length = round(np.mean(subjects_spike_length)) #, dtype=np.int64)
        window_length = round(np.median(subjects_spike_length)) #, dtype=np.int64)
        # print(window_length)
        
        for i in np.arange(len(names)):

                subjects[i].windowing(subjects[i].filt_acc, subjects[i].filt_ang, subjects[i].spikes, window_length)

                unique, counts = np.unique(subjects[i].labels, return_counts=True)
                d = dict(zip(unique, counts))
                print(f'\n  Subject_{subjects[i].tag} name: {subjects[i].name}:')
                print(f'    Negative windows: {d[0.0]}')
                print(f'    Positive windows: {d[1.0]}')
                neg += d[0.0]
                pos += d[1.0]
                ne.append(d[0.0])
                po.append(d[1.0])
                ns.append(d[0.0]//d[1.0])


        print(f'\n  Total:\n    Positives = {pos}\n    Negatives = {neg}')

        return subjects, ns, po, ne


def subjects_init_raw(names, window_length):
        
        print(f'\n# Subjects Before augmentation:')

        subjects = list()
        ns = list()
        po = list()
        ne = list()
        subjects_spike_length = list()
        pos = 0
        neg = 0

        for i in np.arange(len(names)):
                
                n = names[i]

                if i < 9:
                        tg = '0' + str(i+1)
                else:
                        tg = str(i+1)

                subjects.append(recording(name = n, tag = tg))

                subjects[i].read_data()
                subjects[i].filtering()
                subjects[i].windowing(subjects[i].raw_acc, subjects[i].raw_ang, subjects[i].spikes, window_length)

                unique, counts = np.unique(subjects[i].labels, return_counts=True)
                d = dict(zip(unique, counts))
                print(f'\n  Subject_{subjects[i].tag} name: {subjects[i].name}:')
                print(f'    Negative windows: {d[0.0]}')
                print(f'    Positive windows: {d[1.0]}')
                neg += d[0.0]
                pos += d[1.0]
                ne.append(d[0.0])
                po.append(d[1.0])
                ns.append(d[0.0]//d[1.0])


        print(f'\n  Total:\n    Positives = {pos}\n    Negatives = {neg}')

        return subjects, ns, po, ne


# 3. Display base's statistical analysis
def display_base_stats(subjects):

        spikes_time_length = list()
        spikes_sample_length = list()

        for s in subjects:

                spikes_samp = s.spikes[:, 1] - s.spikes[:, 0]
                spikes_sec = spikes_samp / s.fs

                spikes_sample_length.extend(spikes_samp)
                spikes_time_length.extend(spikes_sec)

        # Number of spikes:
        nspks = np.shape(spikes_sample_length)[0]

        # Window length selected as the median/mean of spikes length
        wd_len = round(np.median(spikes_sample_length))
        # wd_len = round(np.mean(spikes_sample_length))

        print(f'''

> Base statistical analysis - {nspks} Spikes recorded

\tMaximum Spike length   ->   {np.max(spikes_time_length):.3f} sec, {np.max(spikes_sample_length):.3f} samples

\tMinimum Spike length   ->   {np.min(spikes_time_length):.3f} sec, {np.min(spikes_sample_length):.3f} samples

\t   Mean Spike length   ->   {np.mean(spikes_time_length):.3f} sec, {np.mean(spikes_sample_length):.3f} samples

\t    Std Spike length   ->   {np.std(spikes_time_length):.3f} sec,  {np.std(spikes_sample_length):.3f} samples

\t Median Spike length   ->   {np.median(spikes_time_length):.3f} sec, {np.median(spikes_sample_length):.3f} samples
  /
\/  Selected window length   ->   {wd_len/64:.3f} sec, {wd_len:.3f} samples
'''.expandtabs(6))


################################################################################################################








########################### SENSOR ROTATION - ARTIFICIAL AUGMENTATION ##################################
##                                                                                                    ##
##                                                                                                    ##
## >> Sensor with its axes:                                                                           ##
##     __________________                                                                             ##
##    |                  |   > Suppose this box as our sensor, and its axes as annotated              ##
##    |           ^ z    |                                                                            ##
##    |          /       |   > Rotation arround the x-axis isn't feasible for our wearable.           ##
##    |   x <---|        |                                                                            ##
##    |         |        |   > We can rotate arround y and z axis, by +/- 10 degrees respectively     ##
##    |       y v        |       the same way a watch can rotate on our wrist.                        ##
##    |__________________|                                                                            ##
##                          ** Z-axis is perpendicular with x and y axis respectively,                ##
##                             pointing upwards.                                                      ##
##                                                                                                    ##
##                                                                                                    ##
########################################################################################################

##### SPIKE-WINDOWS ARTIFICIAL AUGMENTATION ####################################################################
        

# 1. Generate rotation matrix
def rand_rot(theta):
        Ry = np.array([
                [np.cos(theta), 0, np.sin(theta)],
                [0, 1, 0],
                [-np.sin(theta), 0, np.cos(theta)]
        ])
        Rz = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
        ])

        r = rand.randint(1, 4)
        if r == 1:
                R = Ry
        elif r == 2:
                R = Rz
        elif r == 3:
                R = np.dot(Ry,Rz)
        elif r == 4:
                R = np.dot(Rz,Ry)

        # print(f'{np.dot(Ry,Rz)}\n\n{np.dot(Rz,Ry)}\n')

        zer = np.zeros( (3, 3) )
        Q1 = np.vstack( (R, zer) )
        Q2 = np.vstack( (zer, R) )
        Q = np.hstack( (Q1, Q2) )
        # print(Q)

        return Q


# 2. Balance positive and negative class registrations
def balance_windows(subjects, ns, posi, neg):
        newsubj = deepcopy(subjects)
        s_ind = 0
        for s, n, pos, neg in zip(subjects, ns, posi, neg):
                lst = np.round(np.random.normal(loc=0.0, scale=10, size = n+1), 3)
                ind = np.transpose( np.nonzero(s.labels) )
                _, d2, d3 = np.shape(s.windows)
                temp = np.empty( [(n+1)*pos, d2, d3] )
                j = 0
                for theta in lst:
                        # theta = np.pi/2
                        Q = rand_rot(theta)
                        for i in ind:
                                ## (i)  Counter-Clockwise rotation:
                                temp[j] = np.dot(np.squeeze(s.windows[i, :, :]), Q)

                                ## (ii) Clockwise rotation:
                                # temp[j] = np.transpose( np.dot(Q, np.squeeze(np.transpose(s.windows[i, :, :])) ) )
                                
                                j += 1

                aug_end = neg - pos
                temp = temp[:aug_end, :, :]
                # print(f'Before: {np.shape(newsubj[s_ind].windows)}')
                # newsubj[s_ind].extend_windows(temp, n*pos)
                newsubj[s_ind].extend_windows(temp, aug_end)
                s_ind += 1
                # print(f'After: {np.shape(newsubj[s_ind-1].windows)}')

        print(f'\n# After augmentation:')
        neg = 0
        pos = 0
        for nsj in newsubj:
                unique, counts = np.unique(nsj.labels, return_counts=True)
                d = dict(zip(unique, counts))
                print(f'\n  Subject_{nsj.tag}:')
                print(f'    Negative windows: {d[0.0]}')
                print(f'    Positive windows: {d[1.0]}')
                neg += d[0.0]
                pos += d[1.0]
        
        print(f'\n  Total:\n    Positives = {pos}\n    Negatives = {neg}')

        # return newsubj, ind
        return newsubj


################################################################################################################








##### STANDARDIZATION ##########################################################################################


# 1. Calculate standardization parameters
def standardization_parameters(windows):

        means = np.mean(windows, axis = (0,1))
        stds = np.std(windows, axis = (0,1))
        
        return means, stds


# 2. Apply standardization method
def apply_standardization(windows, means, stds):

        for i in np.arange(windows.shape[2]):
                windows[:, :, i] = ( windows[:, :, i] - means[i] ) / stds[i]

        return windows


#################################################################################################################








#### WINDOWS / SIGNAL PROCESSING ################################################################################


# 1. Concatenate attributes of all subjects in one list (windows, labels, tags, timestamps)
def concatenate_windows(subj_list):
        wds = deepcopy(subj_list[0].windows)
        lbl = deepcopy(subj_list[0].labels)
        tg = np.full(np.shape(subj_list[0].labels), subj_list[0].tag)
        tmst = deepcopy(subj_list[0].timestamps)
        # print("1.", np.shape(subj_list[0].windows), np.shape(subj_list[0].labels), np.shape(tg), np.shape(subj_list[0].timestamps))

        for i in np.arange(len(subj_list))[1:]:
                wds = np.append(wds, subj_list[i].windows, axis = 0)
                lbl = np.append(lbl, subj_list[i].labels, axis = 0)
                temp = np.full(np.shape(subj_list[i].labels), subj_list[i].tag)
                tg = np.append(tg, temp, axis = 0)
                tmst = np.append(tmst, subj_list[i].timestamps, axis = 0)
                # print(f'{i+1}. {np.shape(subj_list[i].windows)}, {np.shape(subj_list[i].labels)}, {np.shape(temp)}, {np.shape(subj_list[i].timestamps)}')

        return wds, lbl, tg, tmst


# 2. Hand mirroring  -  <full-signal>
def hand_mirroring_signals(acc, ang):
        acc_x = np.array([
                [-1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]
        ])

        ang_y_z = np.array([
                [1, 0, 0],
                [0, -1, 0],
                [0, 0, -1]
        ])

        acc = acc @ acc_x
        ang = ang @ ang_y_z

        return acc, ang
        

# 3. Hand mirroring  -  <window-per-window>
def hand_mirroring_windows(wds):
        B = np.array([
                [-1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, -1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1]
        ])

        b = wds @ B
        
        if np.array_equal(b.shape, wds.shape):
                return b
        else:
                print("Error on shape!")


##################################################################################################################









######################################## EVALUATION ANALYSIS ###################################################
##                                                                                                            ##
##                                                                                                            ##
##     1. Raw-signal prediction:                                                                              ##
##                                                                                                            ##
##          In this type of prediction, we want to predict 1 positive class per spike.                        ##
##          In order to evaluate our model with this assumption, we have to separate                          ##
##          false-positive predictions in two cases:                                                          ##
##                                                                                                            ##
##            - fp_1: When the window is non-spike [class 0] and we predict it as a spike [class 1]           ##
##                                                                                                            ##
##            - fp_2: When the window is a spike, we have predicted it as a spike, but we                     ##
##                    have already got a positive prediction for the same spike.                              ##
##                                                                                                            ##
##          They are both false-positives, but their decrease method differs                                  ##
##                                                                                                            ##
##                                                                                                            ##
##     2. Windows prediction:                                                                                 ##
##                                                                                                            ##
##         > Evaluation coefficients:                                                                         ##
##                                                                                                            ##
##             - True_Positives   ->  tp                                                                      ##
##                                                                                                            ##
##             - False_Positives  ->  fp                                                                      ##
##                                                                                                            ##
##             - False_Negatives  ->  fn                                                                      ##
##                                                                                                            ##
##             - True_Negatives   ->  tn                                                                      ##
##                                                                                                            ##
##                                                                                                            ##
################################################################################################################


##### EVALUATION METHODS #########################################################################################


# 1. Calculate evaluation coefficients based on windows prediction
def windows_eval_coeffs(testY, predY, pred_peaks):
        tp, fp, fn, tn = 0, 0, 0, 0

        # From predicion_peaks, we can only extract true and false positives
        for p in pred_peaks:
                
                # Got a positive-class prediction - because of peaks() it is for sure == 1
                if predY[p] == 1:

                        # Prediction and actual class are positive
                        if testY[p] == 1:
                                tp += 1
                        
                        # Positive prediction on an actual negative class
                        else:
                                fp += 1

        # false-negatives = actual-peaks - correctly-predicted-peaks
        Test_peaks, _ = find_peaks(testY)
        fn = Test_peaks.shape[0] - tp

        ## The rest of the windows will be the true negatives
        tn = testY.shape[0] - tn - fp - fn
       
#         print(f'''
#  > Spike-per-spike Method:
# \n\tTrue_positives  = {tp}
# \n\tFalse_positives = {fp}
# \n\tFalse_negatives = {fn}
# \n\tTrue_negatives  = {tn}
#         '''.expandtabs(6))

        return tp, fp, fn, tn
        

# 2. Calculate metrics out of a confusion matrix
def calculate_metrics(cm):

        cm_df = pd.DataFrame(cm, 
                columns = ['Predicted Negative', 'Predicted Positive'],
                index = ['Actual Negative', 'Actual Positive'])

        print(cm_df)
        
        tn, fp, fn, tp = cm.ravel()
        
        '''
        > weighted accuracy:

                wacc = (tp_rate + tn_rate) / 2  =  ( (tp / (tp + fn)) + (tn / (tn + fp)) ) / 2

        '''
        # Acc  = (tp + tn) / (tp + tn + fp + fn)
        # Acc  = (tp*wgt + tn) / ((tp + fn)*wgt + fp + tn)      # equal method
        Acc  = ( (tp / (tp + fn)) + (tn / (tn + fp)) ) / 2
        Prec = tp / (tp + fp)
        Rec  = tp / (tp + fn)
        F1s  = 2 * (Prec * Rec) / (Prec + Rec)

        return Acc, Prec, Rec, F1s


##################################################################################################################





##### POST PROCESSING ############################################################################################

# 1. Extract the speed value of true-positive spikes
def speed_calculator(wds, snum):
        spk_speed = []
        pli = 0
        for i in wds:
                spk = i * 9.80665         # g to m/sec^2

                spk_axs_norm = np.linalg.norm(spk, 1, axis = 1) 
                # Max amplitude
                max_y = np.argwhere(spk[:, 1] == np.amax(spk[:, 1])).flatten()
                max_z = np.argwhere(spk[:, 2] == np.amin(spk[:, 2])).flatten()
                #print('\n\tspike: ', pli, '\n')
                #print(max_y, max_z)
                # print(max_y, '\n', max_z)
                max_i = 0
                for my in max_y:
                        for mz in max_z:
                                if my == mz:
                                    max_i = my

                if max_i == 0:
                    samps = spk[max_z, 1]
                    smax = np.argmax(samps)
                    max_i = max_z[smax]
                
                #print(max_i, spk[max_i, 1:])
                #print('\n\n')
                # if there is no local minima before max, from the start of the window,
                # select the first window value/minimum. Else, pick the last minima.
                l_min_0 = argrelextrema(spk[:max_i+1, 2], np.greater)[0]
                if np.shape(l_min_0)[0] == 0:
                    l_min_0 = np.argmax(spk[:max_i+1, 2])
                    # print(l_min_0)
                else:
                    l_min_0 = l_min_0[-1]
                
                #l_min_0 += 1

                # if there is no local minima after max until the end of the window,
                # select the last window value. Else, pick the first minima.
                #l_min_1 = argrelextrema(spk_axs_norm[max_i:], np.less)[0]
                l_min_1 = argrelextrema(spk[max_i:, 2], np.greater)[0]
                if np.shape(l_min_1)[0] == 0:
                    #l_min_1 = max_i + np.argmin(spk_axs_norm[max_i:])
                    l_min_1 = max_i + np.argmax(spk[max_i:, 2])
                    # print(l_min_1)
                else:
                    l_min_1 = max_i + l_min_1[0]
                
                #l_min_1 = max_i + 1
                # print(l_min_0, max_i, l_min_1)

                l_min_0 = max_i - 3
                l_min_1 = max_i + 1

                spk_time = (l_min_1-1 - l_min_0) / 64         # in sec
                #spk_time = (l_min_1-2 - l_min_0) / 64         # in sec
                #print(spk_time)
                #spk_time = spk_time * 1000                  # in msec
                spike_samples = np.arange(start = l_min_0, stop = l_min_1)
                #spike_samples = np.arange(start = l_min_0, stop = max_i)
                '''
                acceleration:
                spike_start : spike_max  ->  c*x , c > 0
                spike_max   : spike_stop -> -c*x , c > 0
                '''
                #print(spk_time)
                # velocity = sum(acceleration) * sec/sample
                max_v_x = np.trapz(spk[ spike_samples, 0 ]) / 64# * spk_time
                max_v_y = 0#np.trapz(spk[ spike_samples, 1 ]) * spk_time
                max_v_z = np.trapz(spk[ spike_samples, 2 ]) / 64# * spk_time
                max_v = np.sqrt(max_v_x**2 + max_v_y**2 + max_v_z**2)
                #max_v = np.trapz(spk_axs_norm[spike_samples]) * spk_time
                max_v = max_v * 60 * 60 / 1000
                #print(l_min_0, max_i, l_min_1)
                
                if pli == snum:
                    fig = plt.figure()
                    plt.plot(spk)
                    plt.ylabel(r'$\dfrac{m}{s^{2}}$', fontsize = 11, labelpad = -10)
                    plt.xlabel('samples', fontsize = 12)
                    plt.grid()
                    plt.axvspan(xmin = l_min_0, xmax = l_min_1, color = 'lightgreen', alpha = 0.5)
                    plt.axvline(max_i, color = 'red', linestyle = 'dashed', linewidth = 2)
                                    
                pli += 1
                spk_speed.append(max_v)
                
        #print(np.shape(spk_speed))
        return spk_speed


def spk_speed(results, n_subjects):

        names = np.asarray(['sltn', 'gali', 'sdrf', 'pasx', 'anti', 'komi', 'fot', 'agge', 'conp', 'LH_galios'])
        subj, _, _, _ = subjects_init_raw(names, 43)
        
        pred_spk_spd = np.zeros(10, dtype = object)
        pred_spk_ind = np.zeros(10, dtype = object)
        true_spk_spd = np.zeros(10, dtype = object)
        true_spk_ind = np.zeros(10, dtype = object)
        
        for s in np.arange(n_subjects):
        
        # Subject initialization
                target = results[s, 0]
                pred = results[s, 1]
                #orig_wds = results[s, 2]
                orig_wds = subj[s].windows

                orig_wds = orig_wds[:, :, :3]
                #print(orig_wds.shape)
                
                #print(orig_wds.shape)
        # Calculate predicted spikes speed
                p, _ = find_peaks(pred[:, 0], distance = 22)
                
                pred_peaks = np.zeros(pred.shape, dtype = np.float64)
                pred_peaks[p] = pred[p]

                # Eliminating false positives
                tp_pred = deepcopy(pred_peaks)
                for i in p:
                        if target[i] == 0:
                                tp_pred[i] = 0

                tp_spk_ind = np.nonzero(tp_pred)[0]
                #print(tp_spk_ind.shape)
                spikes_pred = orig_wds[tp_spk_ind]
                
                # m/s
                spk_speed = np.asarray(speed_calculator(spikes_pred))
                # km/h
                spk_speed = spk_speed * (60 * 60) / 1000
                
                #if s == 2:
                #    plt.plot(spikes_pred[0])

                pred_spk_ind[s] = tp_spk_ind
                pred_spk_spd[s] = spk_speed

        # Calculate original spikes speed
                p, _ = find_peaks(target)

                spk_speed = np.asarray(speed_calculator(orig_wds[p]))
                # km/h
                spk_speed = spk_speed * (60 * 60) / 1000

                true_spk_ind[s] = p
                true_spk_spd[s] = spk_speed

        return true_spk_ind, true_spk_spd, pred_spk_ind, pred_spk_spd



def spk_speed_1(results, name):

        subj, _, _, _ = subjects_init_raw(name, 43)
        
        # Subject initialization
        target = results[0]
        pred = results[1]
        #orig_wds = results[s, 2]
        orig_wds = subj[0].windows

        orig_wds = orig_wds[:, :, :3]
        #print(orig_wds.shape)
                
        #print(orig_wds.shape)
        # Calculate predicted spikes speed
        p, _ = find_peaks(pred[:, 0], distance = 22)
       
        pred_peaks = np.zeros(pred.shape, dtype = np.float64)
        pred_peaks[p] = pred[p]

        # Eliminating false positives
        tp_pred = deepcopy(pred_peaks)
        for i in p:
                if target[i] == 0:
                        tp_pred[i] = 0

        tp_spk_ind = np.nonzero(tp_pred)[0]
        #print(tp_spk_ind)
        #print(tp_spk_ind.shape)
        spikes_pred = orig_wds[tp_spk_ind]
        
        # which spike to plot
        snum = int(float(input(f'\n\t- Select the spike to plot: '.expandtabs(6))))
        # m/s
        spk_speed = np.asarray(speed_calculator(spikes_pred, snum))
        # km/h
        spk_speed = spk_speed * (60 * 60) / 1000
       
        #if s == 2:
        #    plt.plot(spikes_pred[0])

        pred_spk_ind = tp_spk_ind
        pred_spk_spd = spk_speed

        # Calculate original spikes speed
        p, _ = find_peaks(target)
        
        #print(p)
        trg_spk_ind = []
        for i in np.arange(len(p)):
            for j in np.arange(len(tp_spk_ind)):
                if np.abs(p[i] - tp_spk_ind[j]) < 15:
                    trg_spk_ind.append(p[i])
        
        #print(trg_spk_ind)
        ##print(np.asarray(tp_spk_ind))
        trg_spk_ind = np.asarray(trg_spk_ind)
        spikes_target = orig_wds[trg_spk_ind]

        spk_speed = np.asarray(speed_calculator(spikes_target, snum))
        # km/h
        spk_speed = spk_speed * (60 * 60) / 1000

        true_spk_ind = trg_spk_ind 
        true_spk_spd = spk_speed

        #plt.show()

        return true_spk_ind, true_spk_spd, pred_spk_ind, pred_spk_spd



def spk_speed_2(results, name):

        subj, _, _, _ = subjects_init_raw(name, 43)
        
        # Subject initialization
        target = results[0]
        pred = results[1]
        #orig_wds = results[s, 2]
        orig_wds = subj[0].windows

        orig_wds = orig_wds[:, :, :3]
        #print(orig_wds.shape)
                
        #print(orig_wds.shape)
        # Calculate predicted spikes speed
        p, _ = find_peaks(pred[:, 0], distance = 22)
       
        pred_peaks = np.zeros(pred.shape, dtype = np.float64)
        pred_peaks[p] = pred[p]

        # Eliminating false positives
        tp_pred = deepcopy(pred_peaks)
        for i in p:
                if target[i] == 0:
                        tp_pred[i] = 0

        tp_spk_ind = np.nonzero(tp_pred)[0]
        #print(tp_spk_ind)
        #print(tp_spk_ind.shape)
        spikes_pred = orig_wds[tp_spk_ind]
        
        # which spike to plot
        snum = 100
        # m/s
        spk_speed = np.asarray(speed_calculator(spikes_pred, snum))
        # km/h
        spk_speed = spk_speed * (60 * 60) / 1000
       
        #if s == 2:
        #    plt.plot(spikes_pred[0])

        pred_spk_ind = tp_spk_ind
        pred_spk_spd = spk_speed

        # Calculate original spikes speed
        p, _ = find_peaks(target)
        
        #print(p)
        trg_spk_ind = []
        for i in np.arange(len(p)):
            for j in np.arange(len(tp_spk_ind)):
                if np.abs(p[i] - tp_spk_ind[j]) < 15:
                    trg_spk_ind.append(p[i])
        
        #print(trg_spk_ind)
        ##print(np.asarray(tp_spk_ind))
        trg_spk_ind = np.asarray(trg_spk_ind)
        spikes_target = orig_wds[trg_spk_ind]

        spk_speed = np.asarray(speed_calculator(spikes_target, snum))
        # km/h
        spk_speed = spk_speed * (60 * 60) / 1000

        true_spk_ind = trg_spk_ind 
        true_spk_spd = spk_speed

        #plt.show()

        return true_spk_ind, true_spk_spd, pred_spk_ind, pred_spk_spd



def flight_time(results, name):

        subj, _, _, _ = subjects_init_raw(name, 43)
        
        # Subject initialization
        target = results[0]
        pred = results[1]
        pred = np.where(pred < 0.6, 0, 1)
        #orig_wds = results[s, 2]
        #orig_wds = subj[0].windows

        #orig_wds = orig_wds[:, :, :3]
        #print(orig_wds.shape)
        
        spk = subj[0].spikes
        trg_str_stp = []
        prd_str_stp = []
        
        # Select only spike-windows
        trg_spk = np.asarray(np.nonzero(target))
        prd_spk = np.asarray(np.nonzero(pred))
        
        # 1st spike - starting window
        trg_str_stp.append(trg_spk[0])
        prd_str_stp.append(prd_spk[0])
        
        # Save spikes' starting and stopping windows
        for i, j in zip(np.arange(len(trg_spk))[1:-1], np.arange(len(prd_spk))[1:-1]):
            
            if trg_spk[i] - trg_spk[i-1] > 1 or trg_spk[i] - trg_spk[i+1] > 1:
                trg_str_stp.append(trg_spk[i])

            if prd_spk[i] - prd_spk[i-1] > 1 or prd_spk[i+1] - prd_spk[i] > 1:
                prd_str_stp.append(prd_spk[i])

        # last spike - stopping window
        trg_str_stp.append(trg_spk[-1])
        prd_str_stp.append(prd_spk[-1])


        trg_str = []
        trg_stp = []
        prd_str = []
        prd_stp = []
        flagt = False
        flagp = False 
        for i in np.arange(len(target)):
            
            if target[i] == 1 and flagt == False:
                trg_str.append(i)
                flagt = True            
            elif target[i] == 0 and flagt == True:
                trg_stp.append(i-1)
                flagt = False

            if pred[i] == 1  and flagp == False:
                prd_str.append(i)
                flagp = True
            elif pred[i] == 0 and flagp == True:
                prd_stp.append(i-1)
                flagp = False

        trg_stp = np.asarray(trg_stp)
        trg_str = np.asarray(trg_str)
        prd_stp = np.asarray(prd_stp)
        prd_str = np.asarray(prd_str)
        
        inds = []
        for i in np.arange(len(prd_str)):

            if prd_stp[i] - prd_str[i] < 5:
                inds.append(i)

        inds = np.asarray(inds)
        prd_str = np.delete(prd_str, inds)
        prd_stp = np.delete(prd_stp, inds)
        
        #trg_spikes = np.asarray(trg_stp) - np.asarray(trg_str)
        #prd_spikes = np.asarray(prd_stp) - np.asarray(prd_str)
        
        # Reshape as rows: [start, stop]
        #trg_spikes_ss = np.reshape(trg_str_stp, (len(spk), 2))
        #prd_spikes_ss = np.reshape(prd_str_stp, (len(spk), 2))
        #trg_spikes_ss = np.reshape(trg_str_stp, (len(trg_str_stp)/2, 2))
        #prd_spikes_ss = np.reshape(prd_str_stp, (len(prd_str_stp)/2, 2))

        # Number of continuous positive windows, for the same spike
        #trg_spikes = trg_spikes_ss[:, 1] - trg_spikes_ss[:, 0]
        #prd_spikes = prd_spikes_ss[:, 1] - prd_spikes_ss[:, 0]

        return trg_str, trg_stp, prd_str, prd_stp 

