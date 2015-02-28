# coding: utf-8

# #行動認識 実験４
# ###HMMの入力でSVM  
# 学習データ: 0-0, 0-1, 0-3, 0-7, 0-9, 0-12, 1-0, 1-1, 1-2, 1-3, 1-4, 1-5, 1-7  
# テストデータ: 0-2, 0-4, 0-6, 0-8, 0-10, 0-11

import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
import sys, pickle

# #結果保存ファイルを生成
# filename = 'n_cluster_' + str(k) + '_wl1.txt'
# resultfile = open(filename, 'w')
# resultfile.write('n_cluster = %d\n' % k)

#コマンドライン引数の取得
argvs = sys.argv
param1 = int(argvs[1]) #window_length
param2 = float(argvs[2]) #C
param3 = float(argvs[3]) #gamma

print param1, param2, param3

#学習、テストデータの読み込み 
postures_train = pickle.load(open("postures_train.dump", "rb"))
postures_test = pickle.load(open("postures_test.dump", "rb"))
labels_train = pickle.load(open("labels_train.dump", "rb"))
labels_test = pickle.load(open("labels_test.dump", "rb"))

##スライディングウィンドウによる特徴量の生成
#学習データ
len_posvec = postures_train.shape[1]
window_length = param1
features_train = np.empty((len(postures_train)-(window_length-1), len_posvec*window_length))

for i in xrange(window_length-1, len(postures_train)):
    feature_reshaped = np.reshape(postures_train[i-(window_length-1):i+1, :], (1, len_posvec*window_length))
    features_train[i-(window_length-1), :] = feature_reshaped

labels_train = labels_train[window_length-1:]
#print features_train.shape, labels_train.shape

#テストデータ
len_posvec = postures_test.shape[1]
features_test = np.empty((len(postures_test)-(window_length-1), len_posvec*window_length))

for i in xrange(window_length-1, len(postures_test)):
    feature_reshaped = np.reshape(postures_test[i-(window_length-1):i+1, :], (1, len_posvec*window_length))
    features_test[i-(window_length-1), :] = feature_reshaped

labels_test = labels_test[window_length-1:]
#print features_test.shape, labels_test.shape

#SVMの学習、テスト
svm_PosCluster = svm.SVC(C=param2, gamma=param3)
svm_PosCluster.fit(features_train, labels_train)
labels_predicted = svm_PosCluster.predict(features_test)
result = accuracy_score(labels_predicted, labels_test)
print result




