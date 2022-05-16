from copy import deepcopy
from numpy import dtype
from utils import *
from filterfuncs import *

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

	#working properly (?)
	# Rz = np.array([
	# 	[np.cos(theta), np.sin(theta), 0],
	# 	[-np.sin(theta), np.cos(theta), 0],
	# 	[0, 0, 1]
	# ])

	r = rand.randint(1, 4)
	r = 1
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


# names = ['sltn', 'sdrf', 'gali', 'pasx', 'anti', 'komi', 'fot']
names = ['sltn', 'sdrf']
subjects, ns, po, ne = subj_tot(names)
ns = np.array(ns) - 1
base_stats(names)

## Test 01 - Compare a window with its corresponding signal part

# st = round(200*44*0.6)
# fig, axs = plt.subplots(4)
# axs[0].plot(subjects[0].filt_acc[-70:-26, :])
# axs[1].plot(subjects[0].windows[-2, :, 0:3])
# axs[2].plot(subjects[0].filt_ang[-70:-26, :])
# axs[3].plot(subjects[0].windows[-2, :, 3:])
# plt.show()
		

# Rotate sensor - positive class augmentation:

 # - Sensor axis:
 #           
 #          ^ z     
 #         /        Rotation arround the x-axis isn't feasible. We can rotate
 #   x <--|         arround y and z axis, by +/- 10 degrees respectively, the
 #        |         same way a watch can rotate on our wrist.
 #      y v
 #


newsubj = deepcopy(subjects)
s_ind = 0
for s, n, posi, neg in zip(subjects, ns, po, ne):
	lst = np.round(np.random.normal(loc=0.0, scale=10, size = n), 3)
	ind = np.transpose( np.nonzero(s.labels) )
	# print(ind[0, 0])
	d1, d2, d3 = np.shape(s.windows)
	temp = np.empty( [n*posi, d2, d3] )
	j = 0
	for theta in lst:
		theta = np.pi/2
		Q = rand_rot(theta)
		for i in ind:
			## (i)  Counter-Clockwise rotation:
			temp[j] = np.dot(np.squeeze(s.windows[i, :, :]), Q)

			## (ii) Clockwise rotation:
			# temp[j] = np.transpose( np.dot(Q, np.squeeze(np.transpose(s.windows[i, :, :])) ) )
			
			j += 1

	print(f'Before: {np.shape(newsubj[s_ind].windows)}')
	newsubj[s_ind].extend_windows(temp, n*posi)
	s_ind += 1
	print(f'After: {np.shape(newsubj[s_ind-1].windows)}')

# plt.plot(temp[0][:][:])
# plt.show()
first_rotated, _, _ = np.shape(subjects[1].windows)
fig, axs = plt.subplots(5)
print(np.shape(newsubj[1].windows), np.shape(subjects[1].windows))
print(np.count_nonzero(newsubj[1].labels))
print(first_rotated)
axs[0].plot(newsubj[1].windows[ind[0, 0], :, 0:3])
axs[0].legend(["X", "Y", "Z"])
axs[1].plot(subjects[1].windows[ind[0, 0], :, 0:3])
axs[1].legend(["X", "Y", "Z"])
axs[2].plot(newsubj[1].windows[first_rotated, :, 0:3])
axs[2].legend(["X", "Y", "Z"])
# axs[3].plot(newsubj[1].windows[ind[0, 0], :, 0])
# axs[3].plot(newsubj[1].windows[first_rotated, :, 1])
# axs[3].legend(["X_auth", "Y_rot"])
# axs[4].plot(newsubj[1].windows[ind[0, 0], :, 1])
# axs[4].plot(newsubj[1].windows[first_rotated, :, 0])
# axs[4].legend(["Y_auth", "X_rot"])
axs[3].plot(newsubj[1].windows[ind[0, 0], :, 0])
axs[3].plot(newsubj[1].windows[first_rotated, :, 2])
axs[3].legend(["X_auth", "Z_rot"])
axs[4].plot(newsubj[1].windows[ind[0, 0], :, 2])
axs[4].plot(newsubj[1].windows[first_rotated, :, 0])
axs[4].legend(["Z_auth", "X_rot"])
plt.show()

# # line 63: 
# # anti gia append (logw to 3d-array), isws prepei na kanw enan empty sto epithymito megethos 
# # kai na ton gemisw ek neou.
# # 1) Na dokimasw to append se allo arxeio:
# # 		-> Doulevei to append. Gia na min to kanoume se kathe epanalipsi
# #		   mporoume na ftiaksoume enan empty, na ton gemisoume kai na kanoume
# #          ena teliko append stous 2 pinakes
# # 2) Na dokimasw to np.empty se allo arxeio
# # 3) Na ftiaxw ena set_windows() stin class, se periptwsi pou epileksw ton tropo_2

