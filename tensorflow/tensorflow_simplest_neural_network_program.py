# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

# MNISTのデータセットを使う
mnist = tf.keras.datasets.mnist

# データセットをロードする
# アンパックして、それぞれtraine_dataとtest_dataに格納
# train_data:60000個、test_data:10000個
(train_data, train_teacher_labels), (test_data,
                                     test_teacher_labels) = mnist.load_data()

# 正則化
# 0-1の間に分布するように変換
train_data, test_data = train_data / 255.0, test_data / 255.0

# シーケンシャルモデル定義
# 入力層ニューロン数：28x28個
# 中間層ニューロン数：512個、ReLu活性化関数
# ドロップアウト層
# 出力層ニューロン数：10個、ソフトマックス活性化関数、確率へ変換してくれる
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

# モデルのセットアップ
# 最適化アルゴリズム：Adam
# 損失関数：sparse_categorical_crossentropy
# 評価関数：accuracy
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

# 学習
# エポック5回
model.fit(train_data, train_teacher_labels, epochs=5)

# 検証
model.evaluate(test_data, test_teacher_labels)