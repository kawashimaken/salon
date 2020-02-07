# -*- coding: utf-8 -*-

import torch
import torchvision
import torchvision.transforms as transforms
# 全てのニューラルネットワークのベースモジュール
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optimizer
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

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
        # 入力データはconv1に渡して、活性化関数ReLUを適用して、その上、プーリング層を加える
        input_data = self.pool(F.relu(self.conv1(input_data)))
        # 2番目畳み込み層の活性化関数をReLUに指定、その上、プーリング
        # 前の層からきたinput_dataをconv2に渡して、活性化関数ReLUを適用して、その上、プーリング層を加える
        input_data = self.pool(F.relu(self.conv2(input_data)))
        # 前の層からきたinput_dataをフォーマット変換する
        # 自動的に変換する
        input_data = input_data.view(-1, 16 * 5 * 5)
        # 前の層からきたinput_dataをlayer1に渡して、活性化関数ReLUを適用する
        input_data = F.relu(self.layer1(input_data))
        # 前の層からきたinput_dataをlayer2に渡して、活性化関数ReLUを適用する
        input_data = F.relu(self.layer2(input_data))
        # 前の層からきたinput_dataをlayer3(出力層)に渡す
        input_data = self.layer3(input_data)
        # 結果を返す
        return input_data


model = CNN()

# 損失関数は交差エントロピー誤差関数を使う
criterion = nn.CrossEntropyLoss()
# 最適化オプティマイザはSDGを使う、学習率lrは0.001、momentumを0.9に設定する
optimizer = optimizer.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 最大学習回数
MAX_EPOCH = 3

#
for epoch in range(MAX_EPOCH):

    total_loss = 0.0
    for i, data in enumerate(train_data_loader, 0):
        # dataから学習対象データと教師ラベルデータを取り出します
        train_data, teacher_labels = data

        # 計算された勾配情報を削除します
        optimizer.zero_grad()

        # モデルでの予測を計算します
        outputs = model(train_data)

        # lossとwによる微分計算します
        loss = criterion(outputs, teacher_labels)
        loss.backward()

        # 勾配を更新します
        optimizer.step()

        # 誤差を累計します
        total_loss += loss.item()

        # 2000ミニバッチずつ、進捗を表示します
        if i % 2000 == 1999:
            print('学習進捗：[%d, %5d] loss: %.3f' % (epoch + 1, i + 1,
                                                 total_loss / 2000))
            total_loss = 0.0

print('学習完了')

data_iterator = iter(test_data_loader)
images, labels = data_iterator.next()

outputs = model(images)
_, predicted = torch.max(outputs, 1)

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
