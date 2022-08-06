
# 0. Libraries
import numpy as np
import pandas as pd
import sys
import os
import platform
import pickle as pkl
from copy import deepcopy
from math import ceil, floor
from scipy.signal import butter, filtfilt, argrelextrema
from sklearn import tree
from matplotlib import pyplot as plt


# 1. Initializations 

def initializations():
	# Constants:
	sls, clear = system_check()
	FS = 64                         ### Sampling frequency
	F_NYQ = FS / 2                  ### signal frequency - Nyquist theorem

	# Subjects' identification:
	names = np.array(['sltn', 'gali', 'sdrf', 'pasx', 'anti','komi','fot', 'agge', 'conp', 'LH_galios'])

	tags = np.empty(len(names))
	for i in np.arange(len(names)):
		if i < 9:
			tags[i] = '0' + str(i)
		else:
			tags[i] = str(i)

	file_folder = case_selection()

	return sls, clear, FS, F_NYQ, names, tags, file_folder



# 2. Functions

def hl_butter_filter(sign, cut_off, ftype):
	'''
			|---> 'low'   for low-pass filter
	ftype --|
			|---> 'high'  for high-pass filter
	'''

	if ftype != 'high' and ftype != 'low':

		sys.exit("Wrong filter type!")
	
	else:
	
		fc = cut_off / F_NYQ
		b, a = butter(6, fc, ftype, analog = False)
	
		return filtfilt(b, a, sign)




'''
	input  = [signal_object], [names]
	         [signal_object], [names]
		     ...
		     [signal_object], [names]

	output = [concatenated_signal_x]    or   [concatenated_auxiliary_signal]
	         [concatenated_signal_y]
	         [concatenated_signal_z]

'''

def concatenate_lists(data, mde):

	# Initialize total_strm channels with the first subject's stream
	if mde:
		strm = data[0, 0]	# Subject 0, signal[0]
		total_strm = [
			strm[:, 0],		# x channel
			strm[:, 1],		# y channel
			strm[:, 2]		# z channel
		]

		# Iterate for the rest of the subjects
		for d in data[1:]:
			# 3 channels of signal-stream per subject
			strm = d[0]
			total_strm[0] = np.concatenate((total_strm[0], strm[:, 0]), axis = 0)
			total_strm[1] = np.concatenate((total_strm[1], strm[:, 1]), axis = 0)
			total_strm[2] = np.concatenate((total_strm[2], strm[:, 2]), axis = 0)

		total_strm = np.transpose(np.asarray(total_strm))

	else:

		total_strm = data[0, 0]

		for d in data[1:]:
			strm = d[0]
			total_strm = np.concatenate((total_strm, strm), axis = 0)

	# total_strm = np.transpose(np.asarray(total_strm))
	return total_strm


def concatenate_labels(labs):
	
	# Initialize labels
	total_labs = np.asarray(labs[0])
	
	for lb in labs[1:]:
		# labels of d subject
		total_labs = np.append(total_labs, lb, axis = 0)

	total_labs = np.asarray(total_labs)

	return total_labs


def extract_indicators(orig, auxi, labs, e_impact):

	Out_X = np.zeros(6)
	Out_Y = np.zeros(1)
	j = 0

	while j < auxi.shape[0]:
		
		# s_o = orig[j]
		s_a = auxi[j]
		# lbl = labs[j]
		
		# if auxi[j]_L1norm > e_impact, search in the next 2 sec and move 2 sec forward
		if np.greater_equal(s_a, e_impact):
			
			# find next local maxima - 3 axes
			search_area = auxi[j : j+2*64] #np.abs(Orig[j:j+2*64, 0]) + np.abs(Orig[j:j+2*64, 1]) + np.abs(Orig[j:j+2*64, 2])
			maxima = argrelextrema(search_area, np.greater)
			# next maxima's index
			nxt_max_ind = j + maxima[0][0]
			loc_max = orig[nxt_max_ind]
			xmax, ymax, zmax = loc_max

			# find average of swing-movement
			swing_ind = np.arange(start = nxt_max_ind - ceil(0.2*64), stop = nxt_max_ind)
			swing_avrg = np.mean( orig[swing_ind], axis = 0)
			xswing, yswing, zswing = swing_avrg

			indicators = np.concatenate((loc_max, swing_avrg), axis = 0)
			Out_X = np.vstack((Out_X, indicators))
			Out_Y = np.append(Out_Y, labs[nxt_max_ind])

			# jump after the calculated maxima
			j += maxima[0][0] + 1
		
		else:
			# keep searching
			j += 1

	return Out_X[1:], Out_Y[1:]


# 3. Plotting

def print_eimpact(s_auxi, e_impact, mode):
	
	if mode == 10:
		full_auxi = concatenate_lists(s_auxi, 0)
	else:
		full_auxi = s_auxi[mode, 0]
	
	plt.figure('auxi vs e-impact')
	
	plt.title('Auxiliary signal with e-impact threshold', fontsize = '18')
	plt.xticks(fontsize = 'large')
	plt.yticks(fontsize = 'large')
	plt.axhspan(ymin = e_impact, ymax = 13, color = '#40B5AD', alpha = 0.3)

	plt.plot(full_auxi, color = '#0C469C', label = 'auxiliary signal', linewidth = 1)
	plt.axhline(y = e_impact, color = '#40B5AD', linewidth = 2, linestyle = '--', label = 'e-impact threshold')
	plt.axhline(y = 0, color = 'red', linewidth = 2, linestyle = '-', label = 'zero-line')
	
	plt.legend( loc = 'upper left', shadow = True, fontsize = 'large', labelcolor = '#ededed', facecolor = '#536878', borderpad = 1.5)
	plt.show()



# 4. Initialize basic variables

def system_check():
	
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

	return sls, clear


def case_message():
	print(f'''\n
> Select spike annotation method:

\t1. WITH slow-shots and WITH jump-services

\t2. WITH slow-shots but WITHOUT jump_services

\t3. WITHOUT slow-shots but WITH jump-services

\t4. WITHOUT slow-shots and WITHOUT jump-services\n'''.expandtabs(6))
        
	case = int(input(f'  Please, type only the case number (1-4): '))
	print(' ')

	return case


def case_selection():
	flag = False
	while not flag:
		case_number = case_message()
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

	return file_folder



sls, clear, FS, F_NYQ, names, tags, file_folder = initializations()






########################################################################################################
' Notes ' ##############################################################################################
########################################################################################################

''' 
__________________________________________________________________________________________

(1) Signal-data structure  ->  s_orig[10], s_auxi[10], tr_data[9]
__________________________________________________________________________________________

                                                                      |- 0 --> channel x
                                                               (:)    |  
                                                         +-> stream +-|- 1 --> channel y
                                       |- 0 --> signal -/             |   
					    (0 - 10/9)     |                              |- 2 --> channel z
    >> tr_data -----> num_of_subject +-|
	                                   |      
	                                   |- 1 --> name 
	
	    Examples:

		    - 1st subject's signal of channel x:        tr_data[0, 0][:, 0]

			- 5th subject's signal 0f channels y, z:    tr_data[4, 0][:, 2]
		    
			- 9th subject's signal of all channels:     tr_data[8, 0][:, :] ~ tr_data[8, 0]
		    
			- 3rd subject's name:                       tr_data[2, 1]



__________________________________________________________________________________________

(2) Spike-data structure  ->  spikes_exact[10], tr_spikes[9]
__________________________________________________________________________________________


			    								   (:)
                          (0 - 10/9)     |- 0 --> spikes (differs between subjects)
    >> tr_spikes -----> num_of_subject +-|
                                         |- 1 --> name
	
	
	    Examples: 				
	
		    - 1st subject's 10th spike:       tr_spikes[0, 0][9]
		    
            - 5th subject's 9-12th spikes:    tr_spikes[0, 0][8:12]

			- 9th subject's spikes:           tr_spikes[8, 0]
		    
			- 3rd subject's name:             tr_spikes[2, 1]



__________________________________________________________________________________________

(3)
__________________________________________________________________________________________

'''