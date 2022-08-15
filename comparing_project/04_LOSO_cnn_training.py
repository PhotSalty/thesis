from utils_case import *
from collections import Counter
from sklearn.utils import class_weight

# with open('tree_out_data.pkl', 'rb') as f:
with open('dt_training_data.pkl', 'rb') as f:
    trn_wds = pkl.load(f)
    trn_lbl = pkl.load(f)
    trn_ind = pkl.load(f)

# with open('dt_testing_data.pkl', 'rb') as f:
#     tst_orig = pkl.load(f)
#     tst_auxi = pkl.load(f)
#     tst_lbls = pkl.load(f)
#     dt_model_collection = pkl.load(f)
#     e_impact = pkl.load(f)


# print(trn_wds.shape, trn_lbl.shape, trn_ind.shape, tst_orig.shape, tst_auxi.shape, tst_lbls.shape)
print(trn_wds.shape, trn_lbl.shape, trn_ind.shape)


for s in np.arange(10):

    cnn_wds_tot = trn_wds[s]
    cnn_lbl_tot = trn_lbl[s]
    cnn_ind_tot = trn_ind[s]

    # cnn_tst_orig = tst_orig[s]
    # cnn_tst_auxi = tst_auxi[s]
    # cnn_tst_lbls = tst_lbls[s]
    
    # print(cnn_wds_tot.shape, cnn_tst_orig.shape)
    print(cnn_wds_tot.shape)

    model_path = 'Output' + sls + 'out' + str(s) + sls + 'Models' + sls
    figure_path = 'Output' + sls + 'out' + str(s) + sls + 'Training_figures' + sls
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    if not os.path.exists(figure_path):
        os.makedirs(figure_path)

    for i in np.arange(cnn_wds_tot.shape[0]):

        cnn_val_wds = cnn_wds_tot[i]
        cnn_val_lbl = cnn_lbl_tot[i]
        cnn_val_ind = cnn_ind_tot[i]

        cnn_trn_wds = cnn_wds_tot[np.arange(len(cnn_wds_tot)) != i]
        cnn_trn_lbl = cnn_lbl_tot[np.arange(len(cnn_lbl_tot)) != i]
        cnn_trn_ind = cnn_ind_tot[np.arange(len(cnn_lbl_tot)) != i]

        cnn_trn_X, cnn_trn_Y, cnn_trn_ind = concatenate_subj_windows(cnn_trn_wds, cnn_trn_lbl, cnn_trn_ind)

        #if i == 3 and s == 3:
        #    print(cnn_trn_X.shape)
        #    fig, axs = plt.subplots(3)
        #    axs[0].plot(cnn_trn_X[0])
        #    axs[1].plot(cnn_trn_X[10])
        #    axs[2].plot(cnn_trn_X[100])

        # Custon weight computation:
        label_weight = dict(Counter(cnn_trn_Y))
        
        if label_weight[0] > label_weight[1]:
            max_w = label_weight[0]
        else:
            max_w = label_weight[1]

        label_weight.update({0 : max_w / label_weight[0], 1: max_w / label_weight[1]})

        # Weight computation through sklearn:
        #label_weight = class_weight.compute_class_weight(class_weight = 'balanced', classes = np.unique(cnn_trn_Y), y = cnn_trn_Y)

        #label_weight = dict(enumerate(label_weight))
        
        print(f'\n\tTest subj: {s}, Validation subj: {i}, Class distribution: {label_weight}'.expandtabs(4))

        in_shape = cnn_trn_X.shape[1:]
        cnn_model, custom_early_stopping = construct_model(in_shape = in_shape)

        history = cnn_model.fit(
                x = cnn_trn_X,
                y = cnn_trn_Y,
                epochs = 30,
                batch_size = 16,
                class_weight = label_weight,
                validation_data = (cnn_val_wds, cnn_val_lbl),
                verbose = 2,
                callbacks = [custom_early_stopping]
        )

        cnn_model.summary()

        fig, axs = plt.subplots(2)
        fig.suptitle(f'Subject out {s}, validation subject {i}')
        axs[0].plot(history.history['accuracy'])
        axs[0].plot(history.history['val_accuracy'])
        axs[0].set_xlabel('epoch')
        axs[0].set_title('Model accuracy')
        axs[0].legend(['train', 'validation'])

        axs[1].plot(history.history['loss'])
        axs[1].plot(history.history['val_loss'])
        axs[1].set_xlabel('epoch')
        axs[1].set_title('Model loss')
        axs[1].legend(['train', 'validation'])
        
        fig.subplots_adjust(wspace = 0.4, hspace = 0.5)

        model_path_f = model_path + 'M_' + str(s) + '_' + str(i) + '.mdl'
        figure_path_f = figure_path + 'M_' + str(s) + '_' + str(i) + '.pdf'
        
        cnn_model.save(filepath = model_path_f)
        fig.savefig(figure_path_f)
        


        

#plt.show()
