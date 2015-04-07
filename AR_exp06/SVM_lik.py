# coding: utf-8
# 線形PCAで次元削減→ガウシアンHMMで尤度計算→尤度を特徴としてSVM(RBFカーネル)
# パラメータはlenSeq_hmm,n_states,lenSeq_svm以外固定

import numpy as np
from sklearn import svm
from sklearn.hmm import GaussianHMM
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
#from sklearn.preprocessing import scale
import sys
import pickle

# 引数の取得
argvs = sys.argv
N_LABEL = 9
lenSeq_pca = 3
dimPCA = 10
lenSeq_hmm = int(argvs[1])
n_states = int(argvs[2])
lenSeq_svm = int(argvs[3])

print "lenSeq_hmm:%d\nn_states:%d\nlenSeq_svm:%d" % (lenSeq_hmm,n_states,lenSeq_svm)

def slidingWindow(X, L):
    # print "input shape:", X.shape
    dimFeature = X.shape[1]
    features = np.empty((X.shape[0]-(L-1), dimFeature*L))
    for i in xrange(L-1, X.shape[0]):
        feature_reshaped = np.reshape(X[i-(L-1):i+1, :], (1, dimFeature*L))
        features[i-(L-1), :] = feature_reshaped
    # print "output shape:", features.shape
    
    return features

# データの読み込み
lefthand_mocap_train = pickle.load(open("../TUMKitchenDataset/lefthand_mocap_train", "rb"))
lefthand_mocap_test = pickle.load(open("../TUMKitchenDataset/lefthand_mocap_test", "rb"))
labels_train = pickle.load(open("../TUMKitchenDataset/labels_train.dump", "rb"))
labels_test = pickle.load(open("../TUMKitchenDataset/labels_test.dump", "rb"))

# 学習データ、テストデータそれぞれについてPCAの入力(複数フレーム)をつくる
## 学習データのPCA入力
features_pca_train = slidingWindow(lefthand_mocap_train, L=lenSeq_pca)

# ## テストデータのPCA入力
features_pca_test = slidingWindow(lefthand_mocap_test, L=lenSeq_pca)

# 線形PCAで学習データ、テストデータのmocapを次元圧縮
pca = PCA(n_components=dimPCA) # n_components: 圧縮する次元数
## 線形PCAの学習と変換
lefthand_pca_train = pca.fit_transform(features_pca_train)
lefthand_pca_test = pca.fit_transform(features_pca_test)

# PCA用のラベル
labels_pca_train = labels_train[lenSeq_pca-1:]
labels_pca_test = labels_test[lenSeq_pca-1:]
# HMM用のラベル
labels_hmm_train = labels_train[lenSeq_hmm+lenSeq_pca-2:]
labels_hmm_test = labels_test[lenSeq_hmm+lenSeq_pca-2:]

# 次元削減したMocap系列をlenSeq_hmmの長さに分割
pcaSeq_train = [[], [], [], [], [], [], [], [], []] # もっと綺麗に書く
for t in range(lefthand_pca_train.shape[0]-(lenSeq_hmm-1)):
    seq = lefthand_pca_train[t:t+lenSeq_hmm, :]
    pcaSeq_train[labels_pca_train[t]].append(seq)

pcaSeq_test = [] # もっと綺麗に書く
for t in range(lefthand_pca_test.shape[0]-(lenSeq_hmm-1)):
    seq = lefthand_pca_test[t:t+lenSeq_hmm, :]
    pcaSeq_test.append(seq)

# 各行動ごとにGaussianHMMをつくりリストに格納し学習
print "training HMM..."
list_ghmm = []
for i in range(N_LABEL):
	list_ghmm.append(GaussianHMM(n_components=n_states)) #n_components: 隠れ状態数
	list_ghmm[i].fit(pcaSeq_train[i])

lik_train = np.empty((lefthand_pca_train.shape[0]-(lenSeq_hmm-1), N_LABEL)) # 学習データ尤度行列
lik_test = np.empty((len(pcaSeq_test), N_LABEL)) # テストデータ尤度行列

# 学習後のHMMにより尤度計算
print "computing likelihood..."
for i in range(N_LABEL): # どのHMMを使うか
	#学習データ
	countT = 0
	labels_lik_train = [] # 学習用尤度列ラベル
	for ii in range(N_LABEL): # 学習用の各系列の最後のラベルがどれか
		for t_train in range(len(pcaSeq_train[ii])):
			lik_train[countT, i] = list_ghmm[i].score(pcaSeq_train[ii][t_train]) ##←おかしい??
			labels_lik_train.append(ii)
			countT += 1
	# テストデータ
	for t_test in range(len(pcaSeq_test)):
		lik_test[t_test, i] = list_ghmm[i].score(pcaSeq_test[t_test])


# HMMのaccuracyを計算
labels_predicted_hmm = np.zeros(len(lik_test))
# 尤度最大のものを推定とする
for t in range(lik_test.shape[0]):
	labels_predicted_hmm[t] = np.argmax(lik_test[t])
	accuracy_hmm = accuracy_score(labels_predicted_hmm, labels_hmm_test)
print "HMM accuracy=%f" % accuracy_hmm


##############################################
# 尤度を特徴量としたSVM

# 特徴量生成
features_SVMlik_train = slidingWindow(lik_train, L=lenSeq_svm)
features_SVMlik_test = slidingWindow(lik_test, L=lenSeq_svm)

# featuresの長さに合わせるためにラベル列の前を消去
labels_SVMlik_train = labels_hmm_train[lenSeq_svm-1:]
labels_SVMlik_test = labels_hmm_test[lenSeq_svm-1:]

# SVMによる学習、テスト
print "training SVM..."
svm_lik = svm.SVC()
svm_lik.fit(features_SVMlik_train, labels_SVMlik_train)

labels_predicted_SVMlik = svm_lik.predict(features_SVMlik_test)
accuracy_SVMlik = accuracy_score(labels_predicted_SVMlik, labels_SVMlik_test)

print "SVM_lik accuracy=%f" % accuracy_SVMlik


###############################################
# PCA出力を特徴としたSVM

# 特徴量生成
features_SVMpca_train = slidingWindow(lefthand_pca_train, L=lenSeq_svm)
features_SVMpca_test = slidingWindow(lefthand_pca_test, L=lenSeq_svm)

# ラベル生成
labels_SVMpca_train = labels_pca_train[lenSeq_svm-1:]
labels_SVMpca_test = labels_pca_test[lenSeq_svm-1:]

svm_pca = svm.SVC()
svm_pca.fit(features_SVMpca_train, labels_SVMpca_train)
labels_predicted_SVMpca = svm_pca.predict(features_SVMpca_test)
accuracy_SVMpca = accuracy_score(labels_predicted_SVMpca, labels_SVMpca_test)
print "SVM_pca accuracy=%f" % accuracy_SVMpca


###############################################
# 尤度+PCA出力のハイブリッド特徴量でSVM

# lik_trainとlefthand_pcaを列方向に結合
pcalik_train = np.hstack((lefthand_pca_train[lenSeq_hmm-1:, :], lik_train))
pcalik_test = np.hstack((lefthand_pca_test[lenSeq_hmm-1:, :], lik_test))
# スライディングウィンドウ(長さlenSeq_svm)で特徴量生成
features_SVMhyb_train = slidingWindow(pcalik_train, L=lenSeq_svm)
features_SVMhyb_test = slidingWindow(pcalik_test, L=lenSeq_svm)
# ラベル生成
labels_SVMhyb_train = labels_hmm_train[lenSeq_svm-1:]
labels_SVMhyb_test = labels_hmm_test[lenSeq_svm-1:]
# SVMによる学習、テスト
svm_hyb = svm.SVC()
svm_hyb.fit(features_SVMhyb_train, labels_SVMhyb_train)
labels_predicted_SVMhyb = svm_hyb.predict(features_SVMhyb_test)
accuracy_SVMhyb = accuracy_score(labels_predicted_SVMhyb, labels_SVMhyb_test)
print "SVM_hyb accuracy=%f" % accuracy_SVMhyb


###############################################
# 生mocapデータを特徴としたSVM

# 特徴量生成
features_SVMmocap_train = slidingWindow(lefthand_mocap_train, L=lenSeq_svm)
features_SVMmocap_test = slidingWindow(lefthand_mocap_test, L=lenSeq_svm)
print features_SVMmocap_train.shape, features_SVMmocap_test.shape

# ラベル生成
labels_SVMmocap_train = labels_train[lenSeq_svm-1:]
labels_SVMmocap_test = labels_test[lenSeq_svm-1:]
print len(labels_SVMmocap_train), len(labels_SVMmocap_test)

# SVMによる学習
svm_mocap = svm.SVC()
svm_mocap.fit(features_SVMmocap_train, labels_SVMmocap_train)
labels_predicted_SVMmocap = svm_mocap.predict(features_SVMmocap_test)
accuracy_SVMmocap = accuracy_score(labels_predicted_SVMmocap, labels_SVMmocap_test)
print "SVM_mocap accuracy=%f" % accuracy_SVMmocap


# 結果の保存
result = {"accuracy_SVMmocap":accuracy_SVMmocap, "accuracy_SVMpca":accuracy_SVMpca, \
		  "accuracy_SVMlik":accuracy_SVMlik, "accuracy_SVMhyb":accuracy_SVMhyb, "accuracy_hmm":accuracy_hmm, \
		  "lenSeq_hmm":lenSeq_hmm, "n_states":n_states, "lenSeq_svm":lenSeq_svm, \
		  "lenSeq_pca":lenSeq_pca, "dimPCA":dimPCA}
filename = "result_Lh%dns%dLs%d.dump" % (lenSeq_hmm, n_states, lenSeq_svm)
pickle.dump(result, open(filename,"wb"))
