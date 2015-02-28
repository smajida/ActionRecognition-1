# coding: utf-8

# #行動認識 実験2
# 学習データ: 0-0, 0-1, 0-3, 0-7, 0-9, 0-12, 1-0, 1-1, 1-2, 1-3, 1-4, 1-5, 1-7  
# テストデータ: 0-2, 0-4, 0-6, 0-8, 0-10, 0-11

import numpy as np
from sklearn import svm

#kmeansのクラスタ数を標準入力から受け取る
wl = int(raw_input())
print 'window_length: %d' % wl

#結果保存ファイルを生成
filename = 'window_length_' + str(wl) + '.txt'
resultfile = open(filename, 'w')
resultfile.write('window_length = %d\n' % wl)

# ##データの前処理
# 学習データ(姿勢クラスタ)

###学習MOTIONデータ

# データの読み込み
data = np.genfromtxt('poses_train_exp02.bvh', delimiter=" ")

#左腕部分のデータを取り出す
left_hand_data = data[:, 40:55]


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
#len_target(推定ラベルフレームと現フレームの間隔)を5~30まで変化させSVMの精度を計算
for len_target in [0, 5, 10, 15, 20, 25, 30]:
	str_len_target = 'len_target=%d' % len_target
	print str_len_target
	resultfile.write(str_len_target + '\n')

	len_anglevec = left_hand_data.shape[1]
	window_length = wl
	features_svm_train = np.empty((len(left_hand_data)-window_length-len_target+1, len_anglevec*window_length))

	for i in xrange(window_length-1, len(left_hand_data)-len_target):
	    feature_reshaped = np.reshape(left_hand_data[i-(window_length-1):i+1, :], (1, len_anglevec*window_length))
	    features_svm_train[i-(window_length-1), :] = feature_reshaped


	#featuresの長さに合わせるためにラベル列の前window_length+len_target-1個を消去
	labels_svm_train = labels_transformed[window_length+len_target-1:]

	#print features_svm_train.shape, labels_svm_train.shape


	#テストデータ
	len_anglevec = left_hand_data00.shape[1]
	window_length = wl
	features_svm_test = np.empty((len(left_hand_data00)-window_length-len_target+1, len_anglevec*window_length))

	for i in xrange(window_length-1, len(left_hand_data00)-len_target):
	    feature_reshaped = np.reshape(left_hand_data00[i-(window_length-1):i+1, :], (1, len_anglevec*window_length))
	    features_svm_test[i-(window_length-1), :] = feature_reshaped

	#featuresの長さに合わせるためにラベル列の前window_length+len_target-1個を消去
	labels_svm_test = labels_test_transformed[window_length+len_target-1:]



	svm_rawmocap_rbf = svm.SVC(kernel='rbf', C=1, gamma=0.0001)
	svm_rawmocap_rbf.fit(features_svm_train, labels_svm_train)
	labels_predicted = svm_rawmocap_rbf.predict(features_svm_test)

	#正解率を計算
	count = 0
	for i in range(len(labels_predicted)):
	    if labels_predicted[i] == labels_svm_test[i]:
	        count += 1
	accuracy = count / np.float(len(labels_predicted))
	result = 'SVM with raw mocap, rbf: accuracy=%f' % accuracy
	print result
	resultfile.write(result + '\n')


resultfile.close()