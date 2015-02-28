# coding: utf-8

# #行動認識 実験2
# 学習データ: 0-0, 0-1, 0-3, 0-7, 0-9, 0-12, 1-0, 1-1, 1-2, 1-3, 1-4, 1-5, 1-7  
# テストデータ: 0-2, 0-4, 0-6, 0-8, 0-10, 0-11

import numpy as np
from sklearn.cluster import KMeans
from sklearn import svm, grid_search
import pickle

#kmeansのクラスタ数を標準入力から受け取る
wl = 1
print 'svmraw, window_length: %d' % wl

#結果保存ファイルを生成
filename = 'svmraw_tune.txt'
resultfile = open(filename, 'w')
resultfile.write('window_length_svmraw = %d\n' % wl)

# ##データの前処理
# 学習データ(姿勢クラスタ)

###学習MOTIONデータ

# データの読み込み
data = np.genfromtxt('poses_train_exp02.bvh', delimiter=" ")

#左腕部分のデータを取り出す
left_hand_data = data[:, 40:55]

#kmeansの読み込み
dump_kmeans = open('kmeans_left_50.dump')
k_means = pickle.load(dump_kmeans)

#学習用のクラスタラベル列を得る
postures = k_means.labels_


# 学習データ(ラベル)

#学習ラベルデータの前処理
import csv
f = open('labels_train_exp02.csv')
reader = csv.reader(f)

labels = []

#lefthandの列をlabelsに抽出
for row in reader:
    if row[0] == 'instance':
        pass
    else:
        labels.append(row[1])


#ラベルの数値への変換
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(labels)
labels_transformed = le.transform(labels)


# テストデータ(姿勢クラスタ)

###テストMOTIONデータ

#bvh読み込み
data00 = np.genfromtxt('poses_test_exp02.bvh', delimiter=" ")
left_hand_data00 = data00[:, 40:55]

#学習データの前処理で学習したモデルk_meansでテストデータのposture列を予測. k=20
postures_test = k_means.predict(left_hand_data00)

# テストデータ(ラベル)

#テストデータの正解ラベル前処理
import csv
f_test = open('labels_test_exp02.csv')
reader_test = csv.reader(f_test)

labels_test = []

#righthandの列をlabelsに抽出
for row in reader_test:
    if row[0] == 'instance':
        pass
    else:
        labels_test.append(row[1])

#数値ラベルへの変換
labels_test_transformed = le.transform(labels_test)



##SVM with raw mocap

#SVM学習時の特徴量とする角度ベクトルのスライディングウィンドウ

#学習データ
len_anglevec = left_hand_data.shape[1]
window_length = wl
features_svm_train = np.empty((len(left_hand_data)-(window_length-1), len_anglevec*window_length))

for i in xrange(window_length-1, len(left_hand_data)):
    feature_reshaped = np.reshape(left_hand_data[i-(window_length-1):i+1, :], (1, len_anglevec*window_length))
    features_svm_train[i-(window_length-1), :] = feature_reshaped

#featuresの長さに合わせるためにラベル列の前9個を消去
labels_svm_train = labels_transformed[window_length-1:]

#テストデータ
len_anglevec = left_hand_data00.shape[1]
window_length = wl
features_svm_test = np.empty((len(left_hand_data00)-(window_length-1), len_anglevec*window_length))

for i in xrange(window_length-1, len(left_hand_data00)):
    feature_reshaped = np.reshape(left_hand_data00[i-(window_length-1):i+1, :], (1, len_anglevec*window_length))
    features_svm_test[i-(window_length-1), :] = feature_reshaped

#featuresの長さに合わせるためにラベル列の前9個を消去
labels_svm_test = labels_test_transformed[window_length-1:]


#グリッドサーチ
parameters = {'kernel':['rbf'], 'gamma':[0.0, 10e-3, 10e-4], 'C':np.logspace(-4, 4, 10)}
svr = svm.SVC()
svm_rawmocap_rbf = grid_search.GridSearchCV(svr, parameters, cv=5, n_jobs=-1)
svm_rawmocap_rbf.fit(features_svm_train, labels_svm_train)
labels_predicted = svm_rawmocap_rbf.predict(features_svm_test)

#正解率を計算
count = 0
for i in range(len(labels_predicted)):
    if labels_predicted[i] == labels_svm_test[i]:
        count += 1
accuracy = count / np.float(len(labels_predicted))
result = 'SVM with raw mocap, rbf: accuracy=%f' % accuracy
print svm_rawmocap_rbf.best_estimator_
print result
resultfile.write(result + '\n')
resultfile.close()