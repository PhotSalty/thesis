
from utils import *
from testing_functions import LOSO_testing, LOSO_testing1


epochs = 3
val = int(sys.argv[1])
n_subjects = 10

base_path = os.path.dirname(os.getcwd())

pkl_path = base_path + sls + 'data' + sls + 'pickle_output' + sls + 'raw_data_' + str(n_subjects) + '_' + file_folder + '.pkl'

with open(pkl_path, 'rb') as f:
    windows = pkl.load(f)
    labels = pkl.load(f)
    tag = pkl.load(f)
    timestamps = pkl.load(f)
    means = pkl.load(f)
    stds = pkl.load(f)


mdl_path = base_path + sls + 'Models' + sls + file_folder + sls + 'epochs_' + str(epochs) 

if val == 1:
    mdl_path += '_val-on' + sls
    results = LOSO_testing1(windows, labels, tag, means, stds, epochs, n_subjects, mdl_path)
elif val == 0:
    mdl_path += '_val-off' + sls
    results = LOSO_testing1(windows, labels, tag, means, stds, epochs, n_subjects, mdl_path)
else:
    print('Not a valid input, please try again')


if not os.path.exists(f'pickles{sls}'):
    os.makedirs(f'pickles{sls}')

with open (f'pickles{sls}testing_results2.pkl', 'wb') as f:
    pkl.dump(results, f)
    pkl.dump(n_subjects, f)

# Random predicted spike:
def plot_selected_spike(sbj):

    target = results[sbj, 0]
    pred_y = results[sbj, 1]

    s1_target_strt = np.nonzero(target)[0][0]
    s1_pred_y_strt = np.nonzero(pred_y)[0][0]

    i = s1_target_strt
    while target[i] >= 0.6:
        i += 1
    s1_target_stop = i + 19      # 10 samples after spike end

    i = s1_pred_y_strt
    while pred_y[i] >= 0.6:
        i += 1
    s1_pred_y_stop = i + 19

    s1_target_strt -= 20        # 10 samples before spike start
    s1_pred_y_strt -= 20


    plt.figure('Predicted vs True spike')
    plt.suptitle(f'First spike of {sbj} subject')
    plt.plot(target[s1_target_strt : s1_target_stop], color = 'darkblue')
    plt.plot(pred_y[s1_target_strt : s1_target_stop], color = 'orange', alpha = 0.6)
    #plt.plot(pred_y[s1_pred_y_strt : s1_pred_y_stop], color = 'orange', alpha = 0.6)
    plt.legend(['target', 'prediction'])
    plt.grid()

    plt.show()

# plot_selected_spike(3)
