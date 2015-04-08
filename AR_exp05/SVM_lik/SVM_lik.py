#coding: utf-8
#HMM尤度を特徴量とするSVM

import numpy as np
import scipy.io as sio
from sklearn import svm
from sklearn.metrics import accuracy_score
import sys, pickle

argvs = sys.argv
s = int(argvs[1])
K = int(argvs[2])
T = int(argvs[3])
paramC = float(argvs[4])
paramGamma = float(argvs[5])
lenSeq = int(argvs[6])

n_label = 9

lik_train = sio.loadmat('../compute_lik/lik_train_s%dK%dT%d.mat' % (s,K,T))['lik_train']
lik_test = sio.loadmat('../compute_lik/lik_test_s%dK%dT%d.mat' % (s,K,T))['lik_test']
labels_test = pickle.load(open("../compute_lik/labels_test.dump", "rb"))
# print lik_test.shape, labels_test.shape

#学習用尤度列の生成
lik_train_stacked = lik_train[0, 0]
#lik_train_stacked.shape
for i in range(1,9):
    lik_train_stacked = np.vstack([lik_train_stacked, lik_train[i, 0]])
    
lik_train_stacked.shape

#lik_train用のラベル列を生成
labels_lik_train = np.zeros(len(lik_train[0, 0]))
for i in range(1, 9):
    labels_i = np.zeros(len(lik_train[i, 0]))
    for li in range(len(labels_i)):
        labels_i[li] = i
        
    labels_lik_train = np.r_[labels_lik_train, labels_i]

labels_lik_train.shape



#スライディングウィンドウ
#学習用
features_train = np.empty((len(lik_train_stacked)-(lenSeq-1), n_label*lenSeq))

for i in xrange(lenSeq-1, len(lik_train_stacked)):
    likelihood_reshaped = np.reshape(lik_train_stacked[i-(lenSeq-1):i+1, :], (1, n_label*lenSeq))
#     print likelihood_reshaped.shape, features_train.shape
    features_train[i-(lenSeq-1), :] = likelihood_reshaped

#featuresの長さに合わせるためにラベル列の前9個を消去
labels_train = labels_lik_train[lenSeq-1:]

#テスト用
features_test = np.empty((len(lik_test)-(lenSeq-1), n_label*lenSeq))

for i in xrange(lenSeq-1, len(lik_test)):
    likelihood_reshaped = np.reshape(lik_test[i-(lenSeq-1):i+1, :], (1, n_label*lenSeq))
#     print likelihood_reshaped.shape, features_train.shape
    features_test[i-(lenSeq-1), :] = likelihood_reshaped

#featuresの長さに合わせるためにラベル列の前を消去
labels_test_ = labels_test[lenSeq-1+(T-1):]

svm_lik = svm.SVC(kernel="rbf", C=paramC, gamma = paramGamma)
svm_lik.fit(features_train, labels_train)

labels_predicted = svm_lik.predict(features_test)
print s,K,T,paramC,paramGamma,lenSeq, accuracy_score(labels_predicted, labels_test_)

accuracy = accuracy_score(labels_predicted, labels_test_)
filename = 'accuracy_s%dK%dT%dC%fgamma%flSeq%d' % (s,K,T,paramC,paramGamma,lenSeq)
resultfile = open(filename, 'w')
resultfile.write('%f\n' % accuracy)
resultfile.close()


