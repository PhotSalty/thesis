from utils import *
from testing_functions import LOSO_testing, solo_test
# import matplotlib.patches as patches
from scipy.stats import pearsonr

def find_label(spikes, j, wdlen, s):

    ls = s + wdlen
    flag = True
    e = round(0.2 * 64)

    while flag:
        bound = ls - spikes[j, 1]
        
        if bound < 0:
            flag = False
            label = 0
        elif bound <= e:
            flag = False
            label = 1
        else:
            if j < len(spikes[:, 0]) - 1:
                j+=1
            else:
                flag = False
                label = 0
	
    return label, j


def prepare_oversampled_windows(name, tg):
    subj = recording(name = name, tag = f'0{tg}')

    subj.read_data()
    subj.filtering()


    #### Initializations:

    wdlen = 43	# samples

    acc = subj.raw_acc
    ang = subj.raw_ang

    facc = subj.filt_acc
    fang = subj.filt_ang

    spks = subj.spikes


    #### Windowing:

    sub_len = ceil(len(facc[:, 0]))
    wds = np.zeros([sub_len + 2 , wdlen, 6])
    labels = deepcopy(wds[:, 0, 0])
    timestamps = deepcopy(labels)

    facc_ext = np.vstack((facc, np.zeros([62, 3])))
    fang_ext = np.vstack((fang, np.zeros([62, 3])))

    i, j, s = 0, 0, 0

    while s <= sub_len:

        ls = s + wdlen

        wds[i, :, :3] = deepcopy(facc_ext[s:ls, :])
        wds[i, :, 3:] = deepcopy(fang_ext[s:ls, :])
            
        timestamps[i] = s / 64

        step = 1
        labels[i], j = find_label(spks, j, wdlen, s)

        i += 1
        s = s + step

    zer = np.zeros([wdlen, 6])
    c = 1
    while np.array_equal(wds[-c, :, :], zer):
            c += 1

    wds = wds[:-c, :, :]
    labels = labels[:-c]
    timestamps = timestamps[:-c]

    subj.windows = deepcopy(wds)
    subj.labels = deepcopy(labels)
    subj.timestamps = deepcopy(timestamps)

    return subj


def perf_test(Test_X, Test_Y, val = 0):

    epochs = 3
    n_subjects = 10

    base_path = os.path.dirname(os.getcwd())

    pkl_path = base_path + sls + 'data' + sls + 'pickle_output' + sls + 'raw_data_' + str(n_subjects) + '_' + file_folder + '.pkl'

    with open(pkl_path, 'rb') as f:
            _ = pkl.load(f)
            _ = pkl.load(f)
            _ = pkl.load(f)
            _ = pkl.load(f)
            means = pkl.load(f)
            stds = pkl.load(f)

    fig_path = base_path + sls + 'new_test' + sls + file_folder + sls + 'epochs_' + str(epochs) 
    mdl_path = base_path + sls + 'Models' + sls + file_folder + sls + 'epochs_' + str(epochs) 
    
    if val == 1:
            mdl_path += '_val-on' + sls
    elif val == 0:
            mdl_path += '_val-off' + sls
            
    return solo_test(Test_X, means, stds, mdl_path, tg)


def only_tp_spikes(spk_strt, spk_stop, test_strt, test_stop):

    temp_strt = []
    temp_stop = []
    for i in np.arange(len(spk_strt)):
        for j in np.arange(len(test_strt)):
            if np.abs(spk_strt[i] - test_strt[j]) < 20:
                temp_strt.append(spk_strt[i])
                temp_stop.append(spk_stop[i])

    spk_strt = np.asarray(temp_strt)
    spk_stop = np.asarray(temp_stop)

    return spk_strt, spk_stop


def plot_video_with_pred(video_spike, prd, t):
    fig, ax = plt.subplots()
    fig.suptitle('Actual spikes, extended spikes, predicted spikes')

    #ax1.plot(t, video_gt, color = 'blue')
    #for ps in pred_spike:
    #    ax1.axvspan(xmin = ps[0]/64, xmax = (ps[1]+43)/64, color = 'lightgreen', alpha = 0.8)

    ax.plot(t, prd, color = 'orange')
    for vs in video_spike:
        ax.axvspan(xmin = vs[0]/64, xmax = (vs[1])/64, color = 'lightgreen', alpha = 0.8)

    ax.legend(['predicted', 'video ground-truth'])
    ax.grid()

    plt.show()


###################### MAIN #########################################################################################################

###### Creating subject

names = np.array(['sltn', 'gali', 'sdrf', 'pasx', 'anti', 'komi', 'fot', 'agge', 'conp', 'LH_galios'])

# tg = 2
ft_gtr = []
ft_prd = []

for tg in np.arange(len(names)):

    name = names[tg]

    tg = '0' + str(tg)
    subject = prepare_oversampled_windows(name, tg)

    t = np.arange(len(subject.raw_acc)/64, step = 1/64)

    Test_X = subject.windows
    Test_Y = subject.labels
    tmstps = subject.timestamps

    val = 0

    pred_Y = perf_test(Test_X, Test_Y, val = 0)
    pred_Y = np.where(pred_Y < 0.6, 0, 1)

###### Make pred and test equal to raw_acc length
    temp = np.zeros(subject.raw_acc.shape[0])
    temp[1:] = pred_Y[:, 0]
    pred_Y = temp

    temp = np.zeros(subject.raw_acc.shape[0])
    temp[1:] = Test_Y
    Test_Y = temp


###### Find prediction and extended spikes start-stop
    pspk_strt = []
    pspk_stop = []
    tspk_strt = []
    tspk_stop = []
    flagp = False
    flagt = False
    for i in np.arange(len(pred_Y)):
        if pred_Y[i] == 1 and flagp == False:
            pspk_strt.append(i)
            flagp = True
        elif pred_Y[i] == 0 and flagp == True:
            pspk_stop.append(i-1)
            flagp = False

        if Test_Y[i] == 1 and flagt == False:
            tspk_strt.append(i)
            flagt = True
        elif Test_Y[i] == 0 and flagt == True:
            tspk_stop.append(i-1)
            flagt = False

    pspk_strt = np.asarray(pspk_strt)
    pspk_stop = np.asarray(pspk_stop)
    tspk_strt = np.asarray(tspk_strt)
    tspk_stop = np.asarray(tspk_stop)

    spikes = subject.spikes

    print('\n\n > Predicted shape: ', pspk_strt.shape, '\n > Video Shape: ', spikes.shape, '\n\n')

###### Select only true positive spikes
    ## prediction spikes
    pspk_strt, pspk_stop = only_tp_spikes(pspk_strt, pspk_stop, spikes[:, 0], spikes[:, 1])
    ## video spikes
    #vspk_strt, vspk_stop = only_tp_spikes(spikes[:, 0], spikes[:, 1], pspk_strt, pspk_stop)
    vspk_strt, vspk_stop = spikes[:, 0], spikes[:, 1]

        
###### Concatenate start with stop
    pred_spike = np.vstack((pspk_strt, pspk_stop)).transpose()
    target_spike = np.vstack((tspk_strt, tspk_stop)).transpose()
    video_spike = np.vstack((vspk_strt, vspk_stop)).transpose()

    print('\n\n > Predicted shape: ', pred_spike.shape, '\n > Video Shape: ', video_spike.shape, '\n\n')

    print(' > Start_delay: ', np.abs(pred_spike[:, 0] - video_spike[:, 0]))

###### flight time ground-truth and predicted spikes
    fltime_gtr = (video_spike[:, 1] - video_spike[:, 0]) / 64
    fltime_prd = ((pred_spike[:, 1] + 43) - pred_spike[:, 0]) / 64

    # df = pd.DataFrame(fltime_gtr, columns = ['ground_truth'])
    # df['prediction'] = fltime_prd
    # df['gt_timestamp'] = np.round(video_spike[:, 0] / 64, 2)
    # df['pred_timestamp'] = np.round(pred_spike[:, 0] / 64, 2)
    # df['start_offset_sec'] = np.round((video_spike[:, 0] - pred_spike[:, 0]) / 64, 2)
    # df['gt_sample'] = video_spike[:, 0]
    # df['pred_sample'] = pred_spike[:, 0]
    # df['start_offset_samp'] = video_spike[:, 0] - pred_spike[:, 0]

    # print(' > Flight time:\n\n')
    # print(df)


    ###### Video spikes signal (0 / 1)
    video_gt = np.zeros(len(subject.raw_acc))
    for s in spikes:
        o = np.arange(s[0], s[1])
        video_gt[o] = 1


    ###### [Shift left] or [extend spike] for 43/64sec
    #prd = deepcopy(pred_Y)

    ## Shift left
    #prd[43:] = prd[:-43]

    ## Extend spike time for 43/64 sec
    #for i in pspk_stop:
    #    prd[i:i+43] = 1


    ###### construct pred_signal:
    prd = np.zeros(len(pred_Y))
    for sp in pred_spike:
        i = sp[0]
        j = sp[1] + 43
        prd[i:j] = 1


    ###### plot ground-truth with prediction spikes
    # plot_video_with_pred(video_spike, prd, t)

    ft_gtr.extend(fltime_gtr)
    ft_prd.extend(fltime_prd)



print(ft_gtr.shape, ft_prd.shape)
###### Pearson Correlation coefficients:
pearson_result = pearsonr(fltime_gtr, fltime_prd)

print('\n\nPearson correlation coefficients:\n', pearson_result) 

fig = plt.figure('Pearson')
plt.plot(fltime_gtr, fltime_prd, 'o')