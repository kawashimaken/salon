# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# 必要なモジュールをインポートします

import sklearn
print(sklearn.__version__)

import matplotlib.pyplot as plt
#
from sklearn import datasets

# アヤメのデータを取り込みます
iris = datasets.load_iris()
print(iris.feature_names)

import pandas as pd
pd.DataFrame(iris.data, columns=iris.feature_names)
# 三次元グラフを描くためのツールをインポートします
from mpl_toolkits.mplot3d import Axes3D

# Principal component analysis (PCA)主成分分析
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import numpy as np
import matplotlib.colors as colors
#
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#

from mlxtend.plotting import plot_decision_regions as pdr

# -----------------------------------------------------------------------------


# 全てのデータを表示します
print(iris.data)

# 最初の一つの特徴量
first_one_feature = iris.data[:, :1]
pd.DataFrame(first_one_feature, columns=iris.feature_names[ :1])
# 最初の二つの特徴量を使います（sepal length (cm)	sepal width (cm)）


first_two_features = iris.data[:, :2]
# 最初の二列のデータを表示します（今回使うデータ）
print(first_two_features)

# 0,1,2,3のなかの2,3のデータ
# 最後の二つの特徴量を使います（petal length (cm)	petal width (cm)）
last_two_features = iris.data[:, 2:]
# 最初の二列のデータを表示します（今回使うデータ）
print(last_two_features)

teacher_labels = iris.target
print(teacher_labels)


# 最初の全ての特徴量の入っているアヤメデータです
all_features=iris.data

x_min, x_max = all_features[:, 0].min() , all_features[:, 0].max()
y_min, y_max = all_features[:, 1].min() , all_features[:, 1].max()

plt.figure(2, figsize=(12, 9))
plt.clf()
# 散布図を描画します
plt.scatter(all_features[:, 0], all_features[:, 1], s=300, c=teacher_labels,cmap=plt.cm.Set2,
            edgecolor='darkgray')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.grid(True)

x_min, x_max = all_features[:, 2].min(), all_features[:, 3].max()
y_min, y_max = all_features[:, 2].min(), all_features[:, 3].max()

plt.figure(2, figsize=(12, 9))
plt.clf()

# 描画します
plt.scatter(all_features[:, 2], all_features[:, 3], s=300, c=teacher_labels,cmap=plt.cm.Set2,
            edgecolor='darkgray')
plt.xlabel('Petal length')
plt.ylabel('Petal width')
plt.show()
# -----------------------------------------------------------------------------


# 二番目の三次元のグラフです
# 同じデータですが、３次元で表現されたときの空間構造を観察しましょう
# 最初の三つの主成分分析次元を描画します
#
fig = plt.figure(1, figsize=(12, 9))
#
ax = Axes3D(fig, elev=-140, azim=100)

# 次元削減します
reduced_features = PCA(n_components=3).fit_transform(all_features)
# 散布図を作成します
ax.scatter(reduced_features[:, 0], reduced_features[:, 1], reduced_features[:, 2],c=teacher_labels,
           cmap=plt.cm.Set2, edgecolor='darkgray', s=200)
plt.show()
# -----------------------------------------------------------------------------


iris = datasets.load_iris()

# 例として、最初の二つの特徴量の2次元データで使用
first_two_features = iris.data[:, [0,1]]
teacher_labels=iris.target

# ターゲットはiris virginica以外のもの
# つまり iris setosa (0) と iris versicolor (1) のみを対象とします
# (領域の2分割)
first_two_features = first_two_features[teacher_labels!=2]
teacher_labels = teacher_labels[teacher_labels!=2]

# 分類用にサポートベクトルマシン (Support Vector Classifier) を用意します
model = SVC(C=1.0, kernel='linear')
# 最初の二つの特徴量（萼片の長さと幅）を「学習」させます
model.fit(first_two_features, teacher_labels)
# 回帰係数
print(model.coef_)
# 切片 (誤差)
print(model.intercept_)

# figureオブジェクト作成サイズを決めます
fig, ax = plt.subplots(figsize=(12,9))

#-------------------------------------------------------------------------------
# 花のデータを描画します
# iris setosa (y=0) のデータのみを取り出す
setosa = first_two_features[teacher_labels==0]
# iris versicolor (y=1) のデータのみを取り出す
versicolor = first_two_features[teacher_labels==1]
# iris setosa のデータ(白い丸)を描画します
plt.scatter(setosa[:,0], setosa[:,1], s=300, c='white', linewidths=0.5, edgecolors='lightgray')
# iris versicolor のデータ（浅い赤い丸）を描画します
plt.scatter(versicolor[:,0], versicolor[:,1], s=300, c='firebrick', linewidths=0.5, edgecolors='lightgray')
#-------------------------------------------------------------------------------
# 回帰直線を描画します
# グラフの範囲を指定します
Xi = np.linspace(4, 7.25)
# 超平面（線）を描画します
Y = -model.coef_[0][0] / model.coef_[0][1] * Xi - model.intercept_ / model.coef_[0][1]

# グラフに線描画します
ax.plot(Xi, Y, linestyle='dashed', linewidth=3)

plt.show()

# -----------------------------------------------------------------------------



iris = datasets.load_iris()

# 例として、3,4番目の特徴量の2次元データで使用
last_two_features = iris.data[:, [2,3]]
#クラスラベルを取得(教師ラベル)
teacher_labels = iris.target

# トレーニングデータとテストデータに分けます
# 今回は訓練データを80%、テストデータは20%とします
# 乱数を制御するパラメータ random_state は None にすると毎回異なるデータを生成するようになります
train_features, test_features, train_teacher_labels, test_teacher_labels = train_test_split(last_two_features , teacher_labels, test_size=0.2, random_state=None )

# データの標準化処理
sc = StandardScaler()
sc.fit(train_features)

# 標準化された特徴量学習データと検証データ
train_features_std = sc.transform(train_features)
test_features_std = sc.transform(test_features)
from sklearn.svm import SVC
# 線形SVMのインスタンスを生成
model = SVC(kernel='linear', random_state=None)

# モデルを学習させます
model.fit(train_features_std, train_teacher_labels)
from sklearn.metrics import accuracy_score

# 学習済モデルに学習データを分類させるときの精度
predict_train = model.predict(train_features_std)
# 分類精度を計算して表示します。
accuracy_train = accuracy_score(train_teacher_labels, predict_train)
print('学習データに対する分類精度： %.2f' % accuracy_train)
# 学習済モデルにテストデータを分類させるときの精度
predict_test = model.predict(test_features_std)
accuracy_test = accuracy_score(test_teacher_labels, predict_test)
#
print('テストデータに対する分類精度： %.2f' % accuracy_test)



#学習と検証用の特徴量データと教師データをそれぞれ結合させます
combined_features_std = np.vstack((train_features_std, test_features_std))
combined_teacher_labels = np.hstack((train_teacher_labels, test_teacher_labels))

fig = plt.figure(figsize=(12,8))

# 散布図関連設定
scatter_kwargs = {'s': 300, 'edgecolor': 'white', 'alpha': 0.5}
contourf_kwargs = {'alpha': 0.2}
scatter_highlight_kwargs = {'s': 200, 'label': 'Test', 'alpha': 0.7}
#
pdr(combined_features_std, combined_teacher_labels, clf=model, scatter_kwargs=scatter_kwargs,
                      contourf_kwargs=contourf_kwargs,
                      scatter_highlight_kwargs=scatter_highlight_kwargs)
plt.show()
# -----------------------------------------------------------------------------
test_data=np.array([[4.1,5.2]])
print(test_data)
test_result = model.predict(test_data)
print(test_result)



# -----------------------------------------------------------------------------

# 分類用にサポートベクトルマシンを用意します
model = SVC(C=1.0, kernel='linear', decision_function_shape='ovr')

all_features=iris.data
teacher_labels=iris.target

# 「学習」させます
model.fit(all_features, teacher_labels)

# データを分類器に与え、分類(predict)させます
result = model.predict(all_features)

print('教師ラベル')
print(teacher_labels)
print('機械学習による分類(predict)')
print(result)

# データ数をtotalに格納します
total = len(all_features)
# ターゲット（正解）と分類(predict)が一致した数をsuccessに格納します
success = sum(result==teacher_labels)

# 正解率をパーセント表示します
print('正解率')
print(100.0*success/total)


test_data=all_features[:1,:]
print(test_data)
test_result = model.predict(test_data)
print(test_result)

test_data=np.array([[2,3,4.1,5.2]])
print(test_data)
test_result = model.predict(test_data)
print(test_result)
