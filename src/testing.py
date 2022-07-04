from utils import *
from keras import backend as K
from keras.models import load_model
from scipy.signal import find_peaks

p = os.path.dirname(os.getcwd())
p1 = p + sls + 'data' + sls + 'pickle_output' + sls
datapkl = p1 + 'full_data.pkl'

with open(datapkl, 'rb') as f:
	windows = pkl.load(f)
	labels = pkl.load(f)
	tag = pkl.load(f)
	timestamps = pkl.load(f)
	means = pkl.load(f)
	stds = pkl.load(f)

## Subject list by their tags
subjects = np.unique(tag)

s = subjects[0]
test_ind = np.asarray(np.where(tag == s))[0]

Test_X = windows[test_ind[0]:test_ind[-1]+1, :, :]
Test_Y = labels[test_ind[0]:test_ind[-1]+1]

Test_X = apply_stadardization(windows = Test_X, means = means, stds = stds)
print(f'\n################################ Test Data ready ################################################')

epochs = 2

model_path = p + sls + 'Models' + sls + 'M' + s + '_epochs_' + str(epochs) + '.mdl'
model = load_model(model_path)

pred_Y = model.predict(x = Test_X)

threshold = 0.8
pred_Y[np.where(pred_Y < threshold)] = 0
pred_Y[np.where(pred_Y >= threshold)] = 1


# Find the local maxima between the predicted values
p, _ = find_peaks(pred_Y[:, 0], distance = 10)

# Plot the comparison of the predictions with the ground truths
plt.figure()
plt.suptitle(f'Subject-{s}:',fontsize=24, y=1)
plt.title(f'Comparing ground-truth labels with predictions',fontsize=16)
plt.plot(Test_Y, color = 'blue', linewidth = 0.5)
plt.plot(pred_Y, color = 'orange', linewidth = 1.5)
plt.plot(p, pred_Y[p, 0], 'x')

# plt.plot(Test_Y, 'o')
# plt.plot(pred_Y, 'x')
plt.show()