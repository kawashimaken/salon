# -*- coding: utf-8 -*-

import torch
import torchvision
import torchvision.transforms as transforms
# 全てのニューラルネットワークのベースモジュール
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optimizer
# -----------------------------------------------------------------------------
# MacOSで、下記のエラーに遭遇する人は、下の二行が必要
# OMP: Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized.
# OMP: Hint This means that multiple copies of the OpenMP runtime have been linked into the program. That is dangerous, since it can degrade performance or cause incorrect results. The best thing to do is to ensure that only a single OpenMP runtime is linked into the process, e.g. by avoiding static linking of the OpenMP runtime in any library. As an unsafe, unsupported, undocumented workaround you can set the environment variable KMP_DUPLICATE_LIB_OK=TRUE to allow the program to continue to execute, but that may cause crashes or silently produce incorrect results. For more information, please see http://www.intel.com/software/products/support/.

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# データフォーマット変更の設定
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 学習データ  train=True
train_data_with_teacher_labels = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform)
train_data_loader = torch.utils.data.DataLoader(
    train_data_with_teacher_labels, batch_size=4, shuffle=True, num_workers=2)
# 検証データ train=False
test_data_with_teacher_labels = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform)
test_data_loader = torch.utils.data.DataLoader(
    test_data_with_teacher_labels, batch_size=4, shuffle=False, num_workers=2)

# クラス名（正解教師ラベル名）
class_names = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
               'ship', 'truck')


# CNNがPyTrochのnn.Module(ニューラルネットワーククラス)を継承する
class CNN(nn.Module):
    def __init__(self):
        '''
        層ごとに定義する、例えば、活性化関数などは、次forward()で定義する
        '''
        super(CNN, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.conv1 = nn.Conv2d(3, 6, 5)
        # 入力チャネル(3)、出力チャネル(6)、カーネル（フィルタ）サイズ(5)、ストライド、パディング、dilation、グループ、バイアスあり、ゼロパディング
        # MaxPooling層（圧縮）
        self.pool = nn.MaxPool2d(2, 2)
        # 畳み込み層(入力チャネル(6)、出力チャネル(16)、カーネルサイズ(5))
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 全結合レイヤー1
        self.layer1 = nn.Linear(16 * 5 * 5, 120)
        # 全結合レイヤー2
        self.layer2 = nn.Linear(120, 84)
        # 全結合レイヤー3 出力層なので、最後は、ニューロン10個、これは分類したいクラスの数となります。
        self.layer3 = nn.Linear(84, 10)

    def forward(self, input_data):
        '''
        ネットワークの（順伝播）の定義（つなげる）
        '''
        # 1番目畳み込み層の活性化関数をReLUに指定、その上、プーリング
        # 入力データはconv1に渡して、活性化関数ReLUを適用して、その上、プーリング層を加えます
        input_data = self.pool(F.relu(self.conv1(input_data)))
        # 2番目畳み込み層の活性化関数をReLUに指定、その上、プーリング
        # 前の層からきたinput_dataをconv2に渡して、活性化関数ReLUを適用して、その上、プーリング層を加えます
        input_data = self.pool(F.relu(self.conv2(input_data)))
        # 前の層からきたinput_dataをフォーマット変換します
        # 自動的に変換する
        input_data = input_data.view(-1, 16 * 5 * 5)
        # 前の層からきたinput_dataをlayer1に渡して、活性化関数ReLUを適用します
        input_data = F.relu(self.layer1(input_data))
        # 前の層からきたinput_dataをlayer2に渡して、活性化関数ReLUを適用します
        input_data = F.relu(self.layer2(input_data))
        # 前の層からきたinput_dataをlayer3(出力層)に渡します
        input_data = self.layer3(input_data)
        # 結果を返す
        return input_data


# モデルのインスタンスを生成します
model = CNN()

# 損失関数は交差エントロピー誤差関数を使います
criterion = nn.CrossEntropyLoss()
# 最適化オプティマイザはSDGを使う、学習率lrは0.001、momentum(慣性項)を0.9に設定します
# momentumが、ここではオプションですが、転がるボールが地面の摩擦抵抗で徐々に減速していくイメージです
optimizer = optimizer.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 最大学習回数
MAX_EPOCH = 1

#
for epoch in range(MAX_EPOCH):

    total_loss = 0.0
    for i, data in enumerate(train_data_loader, 0):
        # dataから学習対象データと教師ラベルデータを取り出します
        train_data, teacher_labels = data

        # 計算された勾配情報を削除（リセット、クリア）します
        optimizer.zero_grad()

        # モデルでの予測を計算します
        outputs = model(train_data)

        # lossとwによる微分計算します
        loss = criterion(outputs, teacher_labels)
        # 勾配を計算します
        loss.backward()
        #
        # 最適化のステップを一回実行します（パラメーターを更新します、たくさんのoptimizerの共通の処理）
        optimizer.step()

        # 誤差を累計します
        total_loss += loss.item()

        # 2000ミニバッチずつ、進捗を表示します
        if i % 2000 == 1999:
            print('学習進捗：[%d, %5d] loss: %.3f' % (epoch + 1, i + 1,
                                                 total_loss / 2000))
            total_loss = 0.0

print('学習完了')

# iter関数を使って、test_data_loaderをiteratorオブジェクトに変換します
data_iterator = iter(test_data_loader)

# iteratorのnext()関数を使って、一つずつ検証データを出して、アンプっくして、それぞれのの変数（imageとlables）に格納します
images, labels = data_iterator.next()

# imagesを学習済みモデルに入力して、その推論結果をoutputsに入れます
outputs = model(images)
print('outputs', outputs)
# 結果：
# tensor([[-0.5670, -0.5408, -0.1798,  1.7548, -1.1307,  1.7946, -0.0550, -0.4681,
#           0.2456, -0.8611],
#         [ 5.9728,  8.5670, -0.7799, -4.4108, -3.3220, -5.0610, -4.5656, -2.8008,
#           6.8845,  1.8749],
#         [ 1.8402,  2.5417, -0.7070, -1.0284, -1.3782, -1.8616, -2.0512, -1.3936,
#           3.4731,  1.0065],
#         [ 3.5036,  1.7275,  0.5213, -2.0472, -0.0412, -3.2352, -2.1648, -2.7098,
#           4.5848,  0.2555]], grad_fn=<AddmmBackward>)
# iteratorから出したデータはバッチになっており
# ouputsも４枚の写真の推論結果配列が入っています

# 一つずつ、推論結果配列の最大値（最も確信しているラベル）取り出します
_, predicted = torch.max(outputs, 1)
print('_', _)
# 結果：それぞれの最大値そのものが入っています
# tensor([1.6123, 5.6203, 3.0886, 3.8317], grad_fn=<MaxBackward0>)
print('predicted', predicted)
# 結果：「最大値は何番目なのか」が入っています
# tensor([3, 9, 1, 0])
print('予測: ', ' '.join('%5s' % class_names[predicted[j]] for j in range(4)))

count_when_correct = 0
total = 0
with torch.no_grad():
    for data in test_data_loader:
        test_data, teacher_labels = data
        results = model(test_data)
        _, predicted = torch.max(results.data, 1)
        total += teacher_labels.size(0)
        count_when_correct += (predicted == teacher_labels).sum().item()

print('検証画像に対しての正解率: %d %%' % (100 * count_when_correct / total))

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
#
with torch.no_grad():
    for data in test_data_loader:
        #
        test_data, teacher_labels = data
        #
        results = model(test_data)
        #
        _, predicted = torch.max(results, 1)
        #
        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.squeeze.html
        c = (predicted == teacher_labels).squeeze()
        #
        for i in range(4):
            label = teacher_labels[i]
            #
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(10):
    print(' %5s クラスの正解率は: %2d %%' % (class_names[i],
                                     100 * class_correct[i] / class_total[i]))
