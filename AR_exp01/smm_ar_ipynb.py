
# coding: utf-8

# #K-meansによる姿勢のクラスタリング

# In[1]:

import numpy as np
from sklearn.cluster import KMeans


# In[2]:

###データの前処理

# データの読み込み
data = np.genfromtxt('poses0_0_motion.bvh', delimiter=" ")
print data
print data.shape


# In[3]:

#右腕部分のデータを取り出す
right_hand_data = data[:, 24:40]
right_hand_data


# In[4]:

#kmeansの実行
k_means = KMeans(n_clusters=20)
k_means.fit(right_hand_data)


# In[5]:

postures = k_means.labels_
print postures
print len(postures)


# In[6]:

#ラベルの前処理
import csv
f = open('labels0_0.csv')
reader = csv.reader(f)

labels = []

#righthandの列をlabelsに抽出
for row in reader:
    #print row
    if row[0] == 'instance':
        pass
    else:
        labels.append(row[2])

print labels[:41]
print len(labels)


# In[7]:

#ラベルの数値への変換
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

le.fit(labels)

print le.classes_

labels_transformed = le.transform(labels)


# In[8]:

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
		n_posture = 20
		n_label = 9


		#出力確率の計算
		n_emit = np.zeros((n_posture, n_label))
		for i in xrange(n_frame):
			n_emit[X[i], y[i]] += 1

		#xについて足す
		xsum_n_emit = np.sum(n_emit, axis=1)

		self.emit_prob = np.zeros((n_posture, n_label))
		for xi in xrange(n_posture):
			for yi in xrange(n_label):
				if n_emit[xi, yi] == 0:
					pass
				else:
					self.emit_prob[xi, yi] = n_emit[xi, yi] / np.float(xsum_n_emit[yi])


		#遷移確率の計算
		n_trans = np.zeros((n_label, n_label))
		for j in xrange(n_frame-1):
			n_trans[y[j], y[j+1]] += 1

		#y_afterについて足す
		yasum_n_trans = np.sum(n_trans, axis=0)

		self.trans_prob = np.zeros((n_label, n_label))
		for y_after in xrange(n_label):
			for y_before in xrange(n_label):
				if n_trans[y_before, y_after] == 0:
					pass
				else:
					self.trans_prob[y_before, y_after] = n_trans[y_before, y_after] / np.float(yasum_n_trans[y_before])

		return self.emit_prob, self.trans_prob

	def predict(self, X):
		"""
		Predict class
		"""
		n_frame = len(X)
		n_label = 9
		self.labels_predicted = np.empty(n_frame)

		#尤度保存用行列
		matP = np.empty((n_frame, n_label))
		#ラベル保存用行列
		matL = np.empty((n_frame, n_label))

		#ヴィタビ経路の計算
		for j in xrange(1, n_frame):
			for yj in xrange(n_label):
				logprob = np.empty(n_label)
				for yk in xrange(n_label):
					#出力確率または遷移確率が0の場合はNone
					if (self.emit_prob[X[j], yj] == 0.) or (self.trans_prob[yj, yk] == 0.):
						logprob[yk] = None
					else:
						logprob[yk] = np.log(self.emit_prob[X[j], yj]) + np.log(self.trans_prob[yj, yk]) + matP[j-1, yk]


				matP[j, yj] = np.max(logprob)
				matL[j, yj] = np.argmax(logprob)
				print matL

		#推定ラベル列の決定
		self.labels_predicted[n_frame-1] = np.argmax(matP[n_frame-1, :])
		for j in reversed(xrange(n_frame-1)):
			self.labels_predicted[j] = matL[j+1, self.labels_predicted[j+1]]

		return self.labels_predicted


# In[9]:

#モーションキャプチャデータで学習
model = Supervised_markov_model()
model.fit(postures, labels_transformed)


# In[10]:

#学習
model.predict(postures)


# In[13]:

print model.labels_predicted[:1240]


# In[ ]:



