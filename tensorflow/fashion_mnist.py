# -*- coding: utf-8 -*-

# 下準備

## TensorFlowのバージョン


import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

"""## Fashion MNISTデータの取得"""

import keras

fashion_mnist = keras.datasets.fashion_mnist

(train_data, train_teacher_labels), (test_data, test_teacher_labels) = fashion_mnist.load_data()

fashion_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

"""## データセットを見る"""

train_data.shape

len(train_teacher_labels)

"""## 検証データの確認"""

test_data.shape

len(test_teacher_labels)

plt.figure()
plt.imshow(train_data[3], cmap='inferno')
plt.colorbar()
plt.grid(False)

"""## データセットの一部を描画する"""

plt.figure(figsize=(12, 12))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_data[i], cmap='inferno')
    plt.xlabel(fashion_names[train_teacher_labels[i]])

"""# 調理手順

## 設定
"""

BATCH_SIZE = 128
NUM_CLASSES = 10
EPOCHS = 20

IMG_ROWS, IMG_COLS = 28, 28

"""## 学習モデルに合わせてデータ調整"""

train_data = train_data.astype('float32')
test_data = test_data.astype('float32')

train_data /= 255
test_data /= 255

print('学習データ　train_data shape:', train_data.shape)
print(train_data.shape[0], 'サンプルを学習します')
print('検証データ　test_data shape:', train_data.shape)
print(test_data.shape[0], 'サンプルを検証します')

"""## 学習モデルの構築"""

from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam

model = Sequential()

# 入力層
model.add(Flatten(input_shape=(IMG_ROWS, IMG_COLS)))
# 中間層
model.add(Dense(128, activation=tf.nn.relu))
# 出力層
model.add(Dense(10, activation=tf.nn.softmax))

model.summary()

"""## モデルのコンパイル"""

model.compile(optimizer=Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


def plot_loss_accuracy_graph(fit_record):
    # 青い線で誤差の履歴をプロットします、検証時誤差は黒い線で
    plt.plot(fit_record.history['loss'], "-D", color="blue", label="train_loss", linewidth=2)
    plt.plot(fit_record.history['val_loss'], "-D", color="black", label="val_loss", linewidth=2)
    plt.title('LOSS')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.show()

    # 緑の線で精度の履歴をプロットします、検証時制度は黒い線で
    plt.plot(fit_record.history['accuracy'], "-o", color="green", label="train_accuracy", linewidth=2)
    plt.plot(fit_record.history['val_accuracy'], "-o", color="black", label="val_accuracy", linewidth=2)
    plt.title('ACCURACY')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc="lower right")
    plt.show()


"""## 学習"""

print('反復学習回数：', EPOCHS)
fit_record = model.fit(train_data, train_teacher_labels,
                       batch_size=BATCH_SIZE,
                       epochs=EPOCHS,
                       verbose=1,
                       validation_data=(test_data, test_teacher_labels))

"""## 学習プロセスのグラフ"""

plot_loss_accuracy_graph(fit_record)

"""## 検証"""

result_score = model.evaluate(test_data, test_teacher_labels)

print('検証誤差:', result_score[0])
print('検証正確率:', result_score[1])

"""## 予測"""

# 検証データから画像を表示します
data_location = 4
img = test_data[data_location]
print(img.shape)

img = (np.expand_dims(img, 0))
print(img.shape)

predictions_result_array = model.predict(img)
print(predictions_result_array)

number = np.argmax(predictions_result_array[0])
print('予測結果：', fashion_names[number])

"""## 学習済モデル保存"""

model.save('keras-fashion-mnist-model.h5')
