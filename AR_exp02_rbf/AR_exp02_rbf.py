# coding: utf-8

# #行動認識 実験2
# 学習データ: 0-0, 0-1, 0-3, 0-7, 0-9, 0-12, 1-0, 1-1, 1-2, 1-3, 1-4, 1-5, 1-7  
# テストデータ: 0-2, 0-4, 0-6, 0-8, 0-10, 0-11

import numpy as np
from sklearn.cluster import KMeans
from sklearn import svm
import pickle

#kmeansのクラスタ数を標準入力から受け取る
k = int(raw_input())
print 'n_cluster: %d' % k

#結果保存ファイルを生成
filename = 'n_cluster_' + str(k) + '_wl10.txt'
resultfile = open(filename, 'w')
resultfile.write('n_cluster = %d\n' % k)

# ##データの前処理
# 学習データ(姿勢クラスタ)

###学習MOTIONデータ

# データの読み込み
data = np.genfromtxt('poses_train_exp02.bvh', delimiter=" ")

#左腕部分のデータを取り出す
left_hand_data = data[:, 40:55]

#kmeansの実行
k_means = KMeans(n_clusters=k, n_jobs=-1)
k_means.fit(left_hand_data)
#繰り返しkmeansを実行しinertia(最近クラスタからの距離和)が小さいものを採用
for i in range(9):
    k_means_new = KMeans(n_clusters=k, n_jobs=-1)
    k_means_new.fit(left_hand_data)
    if k_means_new.inertia_ < k_means.inertia_:
        k_means = k_means_new

#学習後のkmeansのモデルを保存
dumpname = 'kmeans_left_' + str(k) + '.dump'
dumpfile = open(dumpname, 'w')
pickle.dump(k_means, dumpfile)

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


#Suvervised Markov Model class
class Supervised_markov_model:

	def __init__(self):
		"""
		Constructor
		"""
		self.emit_prob = None
		self.trans_prob = None


	def fit(self, X, y):
		"""
		Fitting Model
		"""
		#各定数の設定
		n_frame = len(X)
		n_posture = k
		n_label = len(le.classes_)


		#出力確率の計算
		n_emit = np.zeros((n_posture, n_label))
		n_emit.fill(10**(-10)) #出力確率の要素がゼロにならないように微小値を代入
		for i in xrange(n_frame):
			n_emit[X[i], y[i]] += 1

		#xについて足す
		xsum_n_emit = np.sum(n_emit, axis=0)

		self.emit_prob = np.zeros((n_posture, n_label))
		self.emit_prob.fill
		for xi in xrange(n_posture):
			for yi in xrange(n_label):
				if n_emit[xi, yi] == 0:
					pass
				else:
					self.emit_prob[xi, yi] = n_emit[xi, yi] / np.float(xsum_n_emit[yi])

		#遷移確率の計算
		n_trans = np.zeros((n_label, n_label))
		n_trans.fill(10**(-10))
		for j in xrange(n_frame-1):
			n_trans[y[j], y[j+1]] += 1

		#y_afterについて足す
		yasum_n_trans = np.sum(n_trans, axis=1)

		self.trans_prob = np.zeros((n_label, n_label))
		for y_after in xrange(n_label):
			for y_before in xrange(n_label):
				if n_trans[y_before, y_after] == 0:
					pass
				else:
					self.trans_prob[y_before, y_after] = n_trans[y_before, y_after] / np.float(yasum_n_trans[y_before])


	def predict(self, X):
		"""
		Predict class
		"""
		n_frame = len(X)
		n_label = len(le.classes_)
		self.labels_predicted = np.empty(n_frame, dtype=int)

		#尤度保存用行列
		matP = np.empty((n_frame, n_label))
		#初期確率はクラス0が0.99, その他は当確率とする
		matP[0, 0] = 0.99
		for i in xrange(1,n_label):
			matP[0, i] = (1 - 0.99) / (n_label - 1)

		#ラベル保存用行列
		matL = np.empty((n_frame, n_label))

		#ヴィタビ経路の計算
		for j in xrange(1, n_frame):
			for yj in xrange(n_label):
				prob = np.empty(n_label)
				for yk in xrange(n_label):
					#出力確率または遷移確率が0の場合はNone
					if (self.emit_prob[X[j], yj] == 0.) or (self.trans_prob[yk, yj] == 0.):
						prob[yk] = None
					else:
						prob[yk] = self.emit_prob[X[j], yj] * self.trans_prob[yk, yj] * matP[j-1, yk]

				#logprobが全てnanの場合はnanを返す
				count = 0
				for i in prob:
					if np.isnan(i) == True:
						count += 1

				if count == len(prob):
					matP[j, yj] = None
					matL[j, yj] = None
				else:
					matP[j, yj] = np.nanmax(prob)
					matL[j, yj] = np.nanargmax(prob)

			#クラスごとの確率を足すと1になるように正規化
			matP[j, :] = matP[j, :] / np.sum(matP[j, :])

		self.likelihoods = matP

		#推定ラベル列の決定
		self.labels_predicted[n_frame-1] = np.nanargmax(matP[n_frame-1, :])
		for j in reversed(xrange(n_frame-1)):
			self.labels_predicted[j] = matL[j+1, self.labels_predicted[j+1]]

		return self.labels_predicted


# ##HMM

#HMM学習
model = Supervised_markov_model()
model.fit(postures, labels_transformed)
#テストデータのラベル予測
model.predict(postures_test)
#テスト用の尤度ベクトル列を得る
hmm_likelihoods_test = model.likelihoods

#正解率を計算
count = 0
for i in range(len(model.labels_predicted)):
    if model.labels_predicted[i] == labels_test_transformed[i]:
        count += 1
accuracy = count / np.float(len(model.labels_predicted))
result = 'HMM: accuracy=%f' % accuracy
print result
resultfile.write(result + '\n')

#学習データの尤度ベクトル列を得る
model.predict(postures)
hmm_likelihoods_train = model.likelihoods


# ##SVM with likelihood

#SVM学習時の特徴量とする尤度ベクトルのスライディングウィンドウ

#学習データ
n_label = hmm_likelihoods_train.shape[1]
window_length = 10
features_train = np.empty((len(hmm_likelihoods_train)-(window_length-1), n_label*window_length))

for i in xrange(window_length-1, len(hmm_likelihoods_train)):
    likelihood_reshaped = np.reshape(hmm_likelihoods_train[i-(window_length-1):i+1, :], (1, n_label*window_length))
    features_train[i-(window_length-1), :] = likelihood_reshaped


#featuresの長さに合わせるためにラベル列の前9個を消去
labels_train = labels_transformed[window_length-1:]


#テストデータ

n_label = hmm_likelihoods_test.shape[1]
window_length = 10
features_test = np.empty((len(hmm_likelihoods_test)-(window_length-1), n_label*window_length))

for i in xrange(window_length-1, len(hmm_likelihoods_test)):
    likelihood_reshaped = np.reshape(hmm_likelihoods_test[i-(window_length-1):i+1, :], (1, n_label*window_length))
    features_test[i-(window_length-1), :] = likelihood_reshaped

#featuresの長さに合わせるためにラベル列の前9個を消去
labels_test = labels_test_transformed[window_length-1:]


# #SVM with HMM likelihood, linear
# svm_likelihood = svm.SVC(kernel='linear')
# svm_likelihood.fit(features_train, labels_train)
# labels_predicted_svm_likelihood = svm_likelihood.predict(features_test)

# count = 0
# for i in range(len(labels_predicted_svm_likelihood)):
#     if labels_predicted_svm_likelihood[i] == labels_test_transformed[i]:
#         count += 1
# accuracy = count / np.float(len(labels_predicted_svm_likelihood))
# result = 'SVM with HMM likelihood, linear: accuracy=%f' % accuracy
# print result
# resultfile.write(result + '\n')



#SVM with HMM likelihood, rbf
svm_likelihood_rbf = svm.SVC(kernel='rbf')
svm_likelihood_rbf.fit(features_train, labels_train)
labels_predicted_svm_likelihood_rbf = svm_likelihood_rbf.predict(features_test)

count = 0
for i in range(len(labels_predicted_svm_likelihood_rbf)):
    if labels_predicted_svm_likelihood_rbf[i] == labels_test_transformed[i]:
        count += 1
accuracy = count / np.float(len(labels_predicted_svm_likelihood_rbf))        
result = 'SVM with HMM likelihood, rbf: accuracy=%f' % accuracy
print result
resultfile.write(result + '\n')

# # ##SVM with raw mocap

# #SVM学習時の特徴量とする角度ベクトルのスライディングウィンドウ

# #学習データ
# len_anglevec = left_hand_data.shape[1]
# window_length = 2
# features_svm_train = np.empty((len(left_hand_data)-(window_length-1), len_anglevec*window_length))

# for i in xrange(window_length-1, len(left_hand_data)):
#     feature_reshaped = np.reshape(left_hand_data[i-(window_length-1):i+1, :], (1, len_anglevec*window_length))
#     features_svm_train[i-(window_length-1), :] = feature_reshaped

# #featuresの長さに合わせるためにラベル列の前9個を消去
# labels_svm_train = labels_transformed[window_length-1:]

# #テストデータ
# len_anglevec = left_hand_data00.shape[1]
# window_length = 2
# features_svm_test = np.empty((len(left_hand_data00)-(window_length-1), len_anglevec*window_length))

# for i in xrange(window_length-1, len(left_hand_data00)):
#     feature_reshaped = np.reshape(left_hand_data00[i-(window_length-1):i+1, :], (1, len_anglevec*window_length))
#     features_svm_test[i-(window_length-1), :] = feature_reshaped

# #featuresの長さに合わせるためにラベル列の前9個を消去
# labels_svm_test = labels_test_transformed[window_length-1:]


# # svm_rawmocap = svm.SVC(kernel='linear')
# # svm_rawmocap.fit(features_svm_train, labels_svm_train)
# # labels_predicted = svm_rawmocap.predict(features_svm_test)

# # #正解率を計算
# # count = 0
# # for i in range(len(labels_predicted)):
# #     if labels_predicted[i] == labels_svm_test[i]:
# #         count += 1
# # accuracy = count / np.float(len(labels_predicted)) 
# # result = 'SVM with raw mocap, linear: accuracy=%f' % accuracy
# # print result
# # resultfile.write(result + '\n')

# svm_rawmocap_rbf = svm.SVC(kernel='rbf')
# svm_rawmocap_rbf.fit(features_svm_train, labels_svm_train)
# labels_predicted = svm_rawmocap_rbf.predict(features_svm_test)

# #正解率を計算
# count = 0
# for i in range(len(labels_predicted)):
#     if labels_predicted[i] == labels_svm_test[i]:
#         count += 1
# accuracy = count / np.float(len(labels_predicted))
# result = 'SVM with raw mocap, rbf: accuracy=%f' % accuracy
# print result
# resultfile.write(result + '\n')
resultfile.close()