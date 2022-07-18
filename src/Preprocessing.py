from filterfuncs import base_stats
from utils import *

names = ["sltn", "gali", "sdrf", "pasx", "anti", "komi", "fot", "agge", "conp", "LH_galios"]

## Without left-handed subject
# names = ["sltn", "gali", "sdrf", "pasx", "anti", "komi", "fot", "agge", "conp"]

## Threshold selection
# names = ['sltn', 'gali', 'sdrf', 'anti', 'conp']
# names = ['sltn', 'sdrf']

## Print full-base statistics:
base_stats(names, sls)

## Concatenate subjects:
subjects, ns, po, ne = subj_tot(names)
ns = np.array(ns) - 1

## Balance data - Positive-class frames augmentation:
newsubj, ind = balance_windows(subjects, ns, po, ne)


## Subjects with raw-windows
wds_raw, lbl_raw, tg_raw, tmst_raw = concatenate_windows(subjects)
means_raw, stds_raw = standardization_parameters(wds_raw)

## Subjects with artificially-augmented-windows
wds_aug, lbl_aug, tg_aug, tmst_aug = concatenate_windows(newsubj)
means_aug, stds_aug = standardization_parameters(wds_aug)


# Which standardization-parameters should we use?
means = means_raw
stds = stds_raw

## Constructing path and save pickle:
p = os.path.dirname(os.getcwd())
p1 = p + sls + 'data' + sls + 'pickle_output' + sls

data_raw = p1 + 'raw_data' + str(len(names)) + '.pkl'
data_aug = p1 + 'aug_data' + str(len(names)) + '.pkl'

with open(data_raw, 'wb') as f:
        pkl.dump(wds_raw, f)
        pkl.dump(lbl_raw, f)
        pkl.dump(tg_raw, f)
        pkl.dump(tmst_raw, f)
        pkl.dump(means, f)
        pkl.dump(stds, f)

with open(data_aug, 'wb') as f:
        pkl.dump(wds_aug, f)
        pkl.dump(lbl_aug, f)
        pkl.dump(tg_aug, f)
        pkl.dump(tmst_aug, f)
        pkl.dump(means, f)
        pkl.dump(stds, f)


# print(f'\n\nRaw standardization parameters:\n Means = {means_raw}\n Stds = {stds_raw}')
# print(f'\nAugmented standardization parameters:\n Means = {means_aug}\n Stds = {stds_aug}')
# Check pickles:

# print(wds_raw.shape, wds_aug.shape)

# with open(data_raw, 'rb') as f:
#       wds1 = pkl.load(f)
#       lbl1 = pkl.load(f)
#       tg1 = pkl.load(f)
#       tmst1 = pkl.load(f)

# fig, axs = plt.subplots(2)
# axs[0].plot(wds1[300, :, 0])
# axs[1].plot(subjects[0].windows[300, :, 0])

# plt.show()


# def comp(A, B, s):
#       if np.array_equal(A,B):
#               print(f'{s} Yey!')
#       else:
#               print(f'{s} Ney!')

# ss = ['Windows', 'Labels', 'Tags', 'Timestamps']
# comp(wds1, wds, ss[0])
# comp(lbl1, lbl, ss[1])
# comp(tg1, tg, ss[2])
# comp(tmst1, tmst, ss[3])




