# coding: utf-8

# #実験5
# ##HMMのモデル変更 (各行動毎にHMMをつくる)
# 離散姿勢ラベル列のMatlab用学習、テストデータを生成

import numpy as np
#from sklearn.cluster import KMeans
import scipy.io as sio
from sklearn import preprocessing
import sys, csv, pickle

argvs = sys.argv
param1 = int(argvs[1]) #n_cluster
param2 = int(argvs[2]) #lenSeq

###学習データ
#BVHの読み込み
mocap_train = np.genfromtxt('../../TUMKitchenDataset/poses_train.bvh', delimiter=" ")

#学習後のKmeannsモデル読み込み
dumpname = 'kmeans_left_' + str(param1) + '.dump'
kmeans = pickle.load(open(dumpname, 'r'))

#学習姿勢ラベル列を得る
postures_train_num = kmeans.labels_

#学習ラベルデータの読み込み
f_train = open('../../TUMKitchenDataset/labels_train.csv')
reader_train = csv.reader(f_train)

labels_train_str = []

#lefthandの列をlabelsに抽出
for row in reader_train:
    #print row
    if row[0] == 'instance':
        pass
    else:
        labels_train_str.append(row[1])

#学習ラベルデータの数値への変換
le = preprocessing.LabelEncoder()
le.fit(labels_train_str)
labels_train = le.transform(labels_train_str)

#離散姿勢ラベル列をlenSeqの長さに分割

#各行動毎の学習データ保存用リスト
posSeq = [None] * 9

lenSeq = param2
for i in range(len(postures_train_num)-(lenSeq-1)):
    seq = postures_train_num[i:i+lenSeq]
    if posSeq[labels_train[i]] == None: #対応する行動ラベルがNoneならそのまま代入
        posSeq[labels_train[i]] = seq
    else:
        posSeq[labels_train[i]] = np.vstack((posSeq[labels_train[i]], seq)) #行方向に姿勢シーケンスを結合

#各行動毎のシーケンスのサンプル数を確認
# for l in range(9):
#     print posSeq[l].shape

#matファイルにHMMの学習データを保存
filename_train = "posSeq_train_K" + str(param1) + 'T' + str(param2)
sio.savemat(filename_train, {"posSeq_train": posSeq})

###テストデータ

#BVH読み込み
mocap_test = np.genfromtxt('../../TUMKitchenDataset/poses_test.bvh', delimiter=" ")
# print mocap_test.shape
lefthand_mocap_test = mocap_test[:, 40:55]

#学習データの前処理で学習したモデルk_meansでテストデータのposture列を予測.
postures_test_num = kmeans.predict(lefthand_mocap_test)


##離散姿勢ラベル列をlenSeqの長さに分割

#各行動毎のテストデータ保存用リスト
posSeq_test = None

for i in range(len(postures_test_num)-(lenSeq-1)):
    seq = postures_test_num[i:i+lenSeq]
    if posSeq_test == None: #対応する行動ラベルがNoneならそのまま代入
        posSeq_test = seq
    else:
        posSeq_test = np.vstack((posSeq_test, seq)) #行方向に姿勢シーケンスを結合

filename_test = "posSeq_test_K" + str(param1) + 'T' + str(param2)
sio.savemat(filename_test, {"posSeq_test": posSeq_test})
