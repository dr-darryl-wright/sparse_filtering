import pickle
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

from SoftMaxOnline import SoftMaxOnline
from sklearn.metrics import roc_curve
from sklearn import preprocessing
path = "/Users/dew/development/PS1-Real-Bogus/ufldl/sparsefiltering/features/"

data = sio.loadmat("../../data/3pi_20x20_skew2_signPreserveNorm.mat")

#data = sio.loadmat("../../data/md/md_20x20_skew4_SignPreserveNorm_with_confirmed1.mat")
#X = data["X"]
#print np.shape(X)

#y = np.concatenate((data["y"], data["validy"]))
y = data["y"]
print np.shape(y)


features = sio.loadmat(path+"SF_maxiter100_L1_3pi_20x20_skew2_signPreserveNorm_6x6_k400_patches_stl-10_unlabeled_meansub_20150409_psdb_6x6_pooled5.mat")

#features = sio.loadmat(path+"SF_maxiter100_L1_md_20x20_skew4_SignPreserveNorm_with_confirmed1_6x6_k400_patches_stl-10_unlabeled_meansub_20150409_psdb_6x6_pooled5.mat")

pooledFeaturesTrain = features["pooledFeaturesTrain"]
X = np.transpose(pooledFeaturesTrain, (0,2,3,1))
numTrainImages = np.shape(X)[3]
X = np.reshape(X, (int((pooledFeaturesTrain.size)/float(numTrainImages)), \
               numTrainImages), order="F")
scaler = preprocessing.MinMaxScaler()
scaler.fit(X.T)  # Don't cheat - fit only on training data
X = scaler.transform(X.T)
"""
smc = SoftMaxOnline(LAMBDA=0, alpha=0.01, verbose=False, isteps=100)
m, n = np.shape(X)
order = np.random.permutation(m)
X = X[order,:]
y = y[order]

train_x = X[:.75*m,:]
train_y = y[:.75*m]
test_x = X[.75*m:,:]
test_y = y[.75*m:]
m, n = np.shape(train_x)
FoMs = []
test_FoMs = []
for i in range(100):
    print i
    order = np.random.permutation(m)
    if i == 20:
        smc.alpha = 0.001
    if i == 70:
        smc.alpha = 0.0001
    if i == 80:
        smc.alpha = 0.00001
    smc.fit(train_x[order,:],train_y[order])
    pred = smc.predict_proba(train_x)
    indices = np.argmax(pred, axis=1)
    pred = np.max(pred, axis=1)
    pred[indices==0] = 1 - pred[indices==0]
    fpr, tpr, thresholds = roc_curve(train_y, pred)
    FoM = 1-tpr[np.where(fpr<=0.01)[0][-1]]
    print "[+] FoM: %.4f" % (FoM)
    FoMs.append(FoM)
    threshold = thresholds[np.where(fpr<=0.01)[0][-1]]
    print "[+] threshold: %.4f" % (threshold)
    pred = smc.predict_proba(test_x)
    indices = np.argmax(pred, axis=1)
    pred = np.max(pred, axis=1)
    pred[indices==0] = 1 - pred[indices==0]
    fpr, tpr, thresholds = roc_curve(test_y, pred)
    FoM = 1-tpr[np.where(fpr<=0.01)[0][-1]]
    print "[+] FoM: %.4f" % (FoM)
    test_FoMs.append(FoM)
    threshold = thresholds[np.where(fpr<=0.01)[0][-1]]
    print "[+] threshold: %.4f" % (threshold)
    
plt.plot(range(100), FoMs)
plt.plot(range(100), test_FoMs)
plt.show()


pickle.dump(smc,open("SoftMaxOnline_test_trained_3pi_epoch100.pkl","w"))
"""
data = sio.loadmat("../../data/3pi_20x20_skew2_signPreserveNorm.mat")
testX = data["testX"]
y = data["y"]
testy = data["testy"]

features = sio.loadmat(path+"SF_maxiter100_L1_3pi_20x20_skew2_signPreserveNorm_6x6_k400_patches_stl-10_unlabeled_meansub_20150409_psdb_6x6_pooled5.mat")

pooledFeaturesTest = features["pooledFeaturesTest"]

    
X = np.transpose(pooledFeaturesTest, (0,2,3,1))
numTestImages = np.shape(X)[3]
X = np.reshape(X, (int((pooledFeaturesTest.size)/float(numTestImages)), \
                   numTestImages), order="F")
testX = scaler.transform(X.T)

#smc = pickle.load(open("SoftMaxOnline_text.pkl","rb"))
#pickle.dump(smc,open("SoftMaxOnline_test_trained_md.pkl","w"))

#smc = pickle.load(open("SoftMaxOnline_test_trained_3pi_epoch100.pkl","rb"))

trained_smc = pickle.load(open("/Users/dew/development/PS1-Real-Bogus/ufldl/sparsefiltering/classifiers/SoftMax_lambda3.000000e-04_SF_maxiter100_L1_3pi_20x20_skew2_signPreserveNorm_6x6_k400_patches_stl-10_unlabeled_meansub_20150409_psdb_6x6_pooled5.pkl","rb"))

smc = SoftMaxOnline(LAMBDA=3e-4, alpha=1e-10, verbose=True, isteps=100)
smc.trainedParams = trained_smc._trainedParams
smc.architecture = trained_smc._architecture

trained_smc = None

pred = smc.predict_proba(testX)
indices = np.argmax(pred, axis=1)
pred = np.max(pred, axis=1)
pred[indices==0] = 1 - pred[indices==0]
fpr, tpr, thresholds = roc_curve(testy, pred)
FoM = 1-tpr[np.where(fpr<=0.01)[0][-1]]
print "[+] FoM: %.4f" % (FoM)
threshold = thresholds[np.where(fpr<=0.01)[0][-1]]
print "[+] threshold: %.4f" % (threshold)

m, n = np.shape(testX)

#print smc.alpha
#smc.alpha = 0.0001
smc.isteps = 10
#print smc.alpha
#print np.shape(testX)
#print np.shape(testy)
#smc.fit(testX, testy)
for i in range(100):
    print "[*] Iteration", i

    order = np.random.permutation(m)
    testX = testX[order,:]
    testy = testy[order]
    smc.fit(testX, testy)

    pred = smc.predict_proba(testX)
    indices = np.argmax(pred, axis=1)
    pred = np.max(pred, axis=1)
    pred[indices==0] = 1 - pred[indices==0]
    fpr, tpr, thresholds = roc_curve(testy, pred)
    FoM = 1-tpr[np.where(fpr<=0.01)[0][-1]]
    print "[+] FoM: %.4f" % (FoM)
    threshold = thresholds[np.where(fpr<=0.01)[0][-1]]
    print "[+] threshold: %.4f" % (threshold)

