# -*- coding: utf-8 -*-

import tensorflow as tf

print(tf.__version__)

import keras

print(keras.__version__)
from keras.models import Sequential
from keras import backend as Keras
from keras.datasets import mnist
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
import numpy as np

BATCH_SIZE = 128
NUM_CLASSES = 10
EPOCHS = 10

IMG_ROWS, IMG_COLS = 28, 28

handwritten_number_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
#
(train_data, train_teacher_labels), (test_data,
                                     test_teacher_labels) = mnist.load_data()
print('ロードしたあと学習データ　train_data shape:', train_data.shape)
print('ロードしたあと検証データ　test_data shape:', test_data.shape)

print('Channel調整変換前　train_data shape:', train_data.shape)
print('Channel調整変換前　test_data shape:', test_data.shape)
#
if Keras.image_data_format() == 'channels_first':
    train_data = train_data.reshape(train_data.shape[0], 1, IMG_ROWS, IMG_COLS)
    test_data = test_data.reshape(test_data.shape[0], 1, IMG_ROWS, IMG_COLS)
    input_shape = (1, IMG_ROWS, IMG_COLS)
else:
    train_data = train_data.reshape(train_data.shape[0], IMG_ROWS, IMG_COLS, 1)
    test_data = test_data.reshape(test_data.shape[0], IMG_ROWS, IMG_COLS, 1)
    input_shape = (IMG_ROWS, IMG_COLS, 1)

print('Channel調整変換後　train_data shape:', train_data.shape)
print('Channel調整変換後　test_data shape:', test_data.shape)

train_data = train_data.astype('float32')
test_data = test_data.astype('float32')

print(test_data)

train_data /= 255
test_data /= 255

print('学習データ　train_data shape:', train_data.shape)
print(train_data.shape[0], 'サンプルを学習します')
print('検証データ　test_data shape:', test_data.shape)
print(test_data.shape[0], 'サンプルを検証します')

# 学習用教師ラベルデータをOne-hotベクトルに変換します
print('Keras変換前学習用教師ラベルデータ　train_teacher_labels shape:',
      train_teacher_labels.shape)
train_teacher_labels = keras.utils.to_categorical(train_teacher_labels,
                                                  NUM_CLASSES)
print('Keras変換後学習用教師ラベルデータ　train_teacher_labels shape:',
      train_teacher_labels.shape)

# 検証用教師ラベルデータをOne-hotベクトルに変換します
print('Keras変換前検証用教師ラベルデータ　test_teacher_labels shape:',
      test_teacher_labels.shape)
print(test_teacher_labels)
test_teacher_labels = keras.utils.to_categorical(test_teacher_labels,
                                                 NUM_CLASSES)
print('Keras変換後検証用教師ラベルデータ　test_teacher_labels shape:',
      test_teacher_labels.shape)
print(test_teacher_labels)

model = Sequential()
model.add(
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(NUM_CLASSES, activation='softmax'))
model.summary()

model.compile(
    optimizer=keras.optimizers.Adadelta(),
    loss=keras.losses.categorical_crossentropy,
    metrics=['accuracy'])

print('学習させる前　train_data shape:', train_data.shape)
print('学習させる前　test_data shape:', test_data.shape)


def plot_loss_accuracy_graph(fit_record):
    # 青い線で誤差の履歴をプロットします、検証時誤差は黒い線で
    plt.plot(
        fit_record.history['loss'],
        "-D",
        color="blue",
        label="train_loss",
        linewidth=2)
    plt.plot(
        fit_record.history['val_loss'],
        "-D",
        color="black",
        label="val_loss",
        linewidth=2)
    plt.title('LOSS')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.show()

    # 緑の線で精度の履歴をプロットします、検証時制度は黒い線で
    plt.plot(
        fit_record.history['acc'],
        "-o",
        color="green",
        label="train_accuracy",
        linewidth=2)
    plt.plot(
        fit_record.history['val_acc'],
        "-o",
        color="black",
        label="val_accuracy",
        linewidth=2)
    plt.title('ACCURACY')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc="lower right")
    plt.show()


def plot_loss_accuracy_graph(fit_record):
    # 青い線で誤差の履歴をプロットします、検証時誤差は黒い線で
    plt.plot(
        fit_record.history['loss'],
        "-D",
        color="blue",
        label="train_loss",
        linewidth=2)
    plt.plot(
        fit_record.history['val_loss'],
        "-D",
        color="black",
        label="val_loss",
        linewidth=2)
    plt.title('LOSS')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.show()

    # 緑の線で精度の履歴をプロットします、検証時制度は黒い線で
    plt.plot(
        fit_record.history['accuracy'],
        "-o",
        color="green",
        label="train_accuracy",
        linewidth=2)
    plt.plot(
        fit_record.history['val_accuracy'],
        "-o",
        color="black",
        label="val_accuracy",
        linewidth=2)
    plt.title('ACCURACY')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc="lower right")
    plt.show()


print('反復学習回数：', EPOCHS)
fit_record = model.fit(
    train_data,
    train_teacher_labels,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=1,
    validation_data=(test_data, test_teacher_labels))
print('fit_record', fit_record)

plot_loss_accuracy_graph(fit_record)
result_score = model.evaluate(test_data, test_teacher_labels, verbose=0)
print('検証誤差:', result_score[0])
print('検証正解率:', result_score[1])


def plot_image(data_location, predictions_array, real_teacher_labels, dataset):
    predictions_array, real_teacher_labels, img = predictions_array[data_location], real_teacher_labels[data_location], \
                                                  dataset[data_location]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img)
    predicted_label = np.argmax(predictions_array)
    # 文字の色：予測結果と実際のラベルと一致する場合は緑、一致しない場合、赤にします
    if predicted_label == real_teacher_labels:
        color = 'green'
    else:
        color = 'red'
    # np.maxはnumpyの関数で、指定した配列の中、最大値を取り出します、ここでは、predictions_arrayの最大値を返します
    plt.xlabel(
        "{} {:2.0f}% ({})".format(
            handwritten_number_names[predicted_label],
            100 * np.max(predictions_array),
            handwritten_number_names[real_teacher_labels]),
        color=color)


def plot_teacher_labels_graph(data_location, predictions_array,
                              real_teacher_labels):
    predictions_array, real_teacher_labels = predictions_array[
        data_location], real_teacher_labels[data_location]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    thisplot = plt.bar(range(10), predictions_array, color="#666666")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[real_teacher_labels].set_color('green')


def convertOneHotVector2Integers(one_hot_vector):
    return [np.where(r == 1)[0][0] for r in one_hot_vector]


# 予測
prediction_array = model.predict(test_data)

print(test_teacher_labels)
print(convertOneHotVector2Integers(test_teacher_labels))

print(test_data.shape)
print(test_data.shape[0])

# 描画のために検証データを変換しておきます
test_data = test_data.reshape(test_data.shape[0], IMG_ROWS, IMG_COLS)

data_location = 77
plt.figure(figsize=(6, 3))
#
plt.subplot(1, 2, 1)
plot_image(data_location, prediction_array,
           convertOneHotVector2Integers(test_teacher_labels), test_data)
#
plt.subplot(1, 2, 2)
plot_teacher_labels_graph(data_location, prediction_array,
                          convertOneHotVector2Integers(test_teacher_labels))
_ = plt.xticks(range(10), handwritten_number_names, rotation=45)

NUM_ROWS = 3
NUM_COLS = 1
NUM_IMAGES = NUM_ROWS * NUM_COLS
#
plt.figure(figsize=(2 * 2 * NUM_COLS + 2, 2 * NUM_ROWS + 4))
plt.subplots_adjust(wspace=0.4, hspace=0.4)
for i in range(NUM_IMAGES):
    #
    plt.subplot(NUM_ROWS, 2 * NUM_COLS, 2 * i + 1)
    plot_image(i, prediction_array,
               convertOneHotVector2Integers(test_teacher_labels), test_data)
    #
    plt.subplot(NUM_ROWS, 2 * NUM_COLS, 2 * i + 2)
    plot_teacher_labels_graph(
        i, prediction_array, convertOneHotVector2Integers(test_teacher_labels))
    _ = plt.xticks(range(10), handwritten_number_names, rotation=45)

# 検証データから画像を表示します
img = test_data[data_location]
print(img.shape)

plt.imshow(img)
img = (np.expand_dims(img, 0))
img = img.reshape(1, IMG_ROWS, IMG_COLS, 1)
print(img.shape)

predictions_result_array = model.predict(img)

print(predictions_result_array)

number = np.argmax(predictions_result_array[0])
print('予測結果：', handwritten_number_names[number])

model.save('keras-mnist-model.h5')
