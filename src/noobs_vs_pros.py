from utils import *

names = np.asarray(['sltn', 'gali', 'sdrf', 'pasx', 'anti', 'komi', 'fot', 'agge', 'conp', 'LH_galios'])

with open(f'pickles{sls}testing_results.pkl', 'rb') as f:
        results = pkl.load(f)
        n_subjects = pkl.load(f)

#true_spk_ind, true_spk_spd, pred_spk_ind, pred_spk_spd = spk_speed(results, n_subjects)

true_spk_ind = np.empty(10, dtype = object)
true_spk_spd = np.empty(10, dtype = object)
pred_spk_ind = np.empty(10, dtype = object)
pred_spk_spd = np.empty(10, dtype = object)

for s in np.arange(len(names)):
    
    name = np.asarray([names[s]])
    tag = s

    tind, tspd, pind, pspd = spk_speed_2(results[s], name)

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
    
    print(f'\n\n Subject {tag+1}:\n')
    print(df)

    true_spk_ind[s] = tind
    true_spk_spd[s] = tspd
    pred_spk_ind[s] = pind
    pred_spk_spd[s] = pspd
    

nb_ind = np.asarray([3, 4, 5, 6, 7, 9])
pr_ind = np.asarray([0, 1, 2, 8])

true_noobs_ind = true_spk_ind[nb_ind]
true_noobs_spd = true_spk_spd[nb_ind]
true_pros_ind = true_spk_ind[pr_ind]
true_pros_spd = true_spk_spd[pr_ind]

pred_noobs_ind = pred_spk_ind[nb_ind]
pred_noobs_spd = pred_spk_spd[nb_ind]
pred_pros_ind = pred_spk_ind[pr_ind]
pred_pros_spd = pred_spk_spd[pr_ind]

with open('spike_speeds.pkl', 'wb') as f:
    pkl.dump(true_spk_ind, f)
    pkl.dump(true_spk_spd, f)
    pkl.dump(pred_spk_ind, f)
    pkl.dump(pred_spk_spd, f)

tind_nb = []
tspd_nb = []
pind_nb = []
pspd_nb = []
for tind, tspd, pind, pspd in zip(true_noobs_ind, true_noobs_spd, pred_noobs_ind, pred_noobs_spd):
    tind_nb.extend(tind)
    tspd_nb.extend(tspd)
    pind_nb.extend(pind)
    pspd_nb.extend(pspd)

tind_pr = []
tspd_pr = []
pind_pr = []
pspd_pr = []
for tind, tspd, pind, pspd in zip(true_pros_ind, true_pros_spd, pred_pros_ind, pred_pros_spd):
    tind_pr.extend(tind)
    tspd_pr.extend(tspd)
    pind_pr.extend(pind)
    pspd_pr.extend(pspd)


tnoobs = pd.DataFrame(tspd_nb, columns = ['speed'])
pnoobs = pd.DataFrame(pspd_nb, columns = ['speed'])
#noobs['index'] = pind_nb

tpros = pd.DataFrame(tspd_pr, columns = ['speed'])
ppros = pd.DataFrame(pspd_pr, columns = ['speed'])
#pros['index'] = pind_pr


dns_tpros = tpros.plot.kde(bw_method = 'silverman').get_lines()[0].get_xydata()
dns_tnoobs = tnoobs.plot.kde(bw_method = 'silverman').get_lines()[0].get_xydata()

dns_ppros = ppros.plot.kde(bw_method = 'silverman').get_lines()[0].get_xydata()
dns_pnoobs = pnoobs.plot.kde(bw_method = 'silverman').get_lines()[0].get_xydata()

plt.close('all')


fig = plt.figure('Density-t')
fig.suptitle('Spike speed estimation from wrist velocity')
plt.title('Using video-annotated spikes')
plt.plot(dns_tpros[:, 0], dns_tpros[:, 1], color = 'red', label = 'pros')
plt.fill_between(dns_tpros[:,0], dns_tpros[:, 1], 0, color = 'red', alpha = 0.4)
plt.plot(dns_tnoobs[:, 0], dns_tnoobs[:, 1], color = 'blue', label = 'noobs')
plt.fill_between(dns_tnoobs[:,0], dns_tnoobs[:, 1], 0, color = 'blue', alpha = 0.4)
plt.xlabel('Spike speed in km/h')
plt.ylabel('density')
plt.legend()

fig = plt.figure('Density-p')
fig.suptitle('Spike speed estimation from wrist velocity')
plt.title('Using predicted spikes')
plt.plot(dns_ppros[:, 0], dns_ppros[:, 1], color = 'red', label = 'pros')
plt.fill_between(dns_ppros[:,0], dns_ppros[:, 1], 0, color = 'red', alpha = 0.4)
plt.plot(dns_pnoobs[:, 0], dns_pnoobs[:, 1], color = 'blue', label = 'noobs')
plt.fill_between(dns_pnoobs[:,0], dns_pnoobs[:, 1], 0, color = 'blue', alpha = 0.4)
plt.xlabel('Spike speed in km/h')
plt.ylabel('density')
plt.legend()


fig, ax = plt.subplots()
fig.suptitle('Spike speed estimation from wrist velocity')
ax.set_title('Using video-annotated spikes')
ax.hist(tspd_nb, bins = 16, density = True, edgecolor = 'darkblue', facecolor = 'darkblue', alpha = 0.6, label = 'amateur athletes')
ax.hist(tspd_pr, bins = 16, density = True, edgecolor = 'darkred', facecolor = 'darkred', alpha = 0.6, label = 'pro/semi-pro athletes')
ax.legend()
ax.set_xlabel('Spike speed in km/h')
ax.set_ylabel('probability')

fig1, ax1 = plt.subplots()
fig1.suptitle('Spike speed estimation from wrist velocity')
ax1.set_title('Using predicted spikes')
ax1.hist(pspd_nb, bins = 16, density = True, edgecolor = 'darkblue', facecolor = 'darkblue', alpha = 0.6, label = 'amateur athletes')
ax1.hist(pspd_pr, bins = 16, density = True, edgecolor = 'darkred', facecolor = 'darkred', alpha = 0.6, label = 'pro/semi-pro athletes')
ax1.legend()
ax1.set_xlabel('Spike speed in km/h')
ax1.set_ylabel('probability')

plt.show()
