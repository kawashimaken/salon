# -*- coding: utf-8 -*-

import torch

print(torch.__version__)

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torch.autograd import Variable
import torch.nn as nn

import torch.optim as optimizer

#
data_folder = '~/data'
BATCH_SIZE = 8

mnist_data = FashionMNIST(
    data_folder, train=True, download=True, transform=transforms.ToTensor())
#
data_loader = DataLoader(mnist_data, batch_size=BATCH_SIZE, shuffle=False)

data_iterator = iter(data_loader)
images, labels = data_iterator.next()

print(len(images))
print(len(labels))

# 学習データ
train_data_with_labels = FashionMNIST(
    data_folder, train=True, download=True, transform=transforms.ToTensor())
train_data_loader = DataLoader(
    train_data_with_labels, batch_size=BATCH_SIZE, shuffle=True)

# 検証データ
test_data_with_labels = FashionMNIST(
    data_folder, train=False, download=True, transform=transforms.ToTensor())
test_data_loader = DataLoader(
    test_data_with_labels, batch_size=BATCH_SIZE, shuffle=True)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 入力層
        self.layer1 = nn.Linear(28 * 28, 100)
        # 中間層（隠れ層）
        self.layer2 = nn.Linear(100, 50)
        # 出力層
        self.layer3 = nn.Linear(50, 10)

    def forward(self, input_data):
        input_data = input_data.view(-1, 28 * 28)
        input_data = self.layer1(input_data)
        input_data = self.layer2(input_data)
        input_data = self.layer3(input_data)
        return input_data


model = MLP()

# ソフトマックスロスエントロピー
lossResult = nn.CrossEntropyLoss()
# SGD
optimizer = optimizer.SGD(model.parameters(), lr=0.01)

# 最大学習回数
MAX_EPOCH = 8

for epoch in range(MAX_EPOCH):
    total_loss = 0.0
    for i, data in enumerate(train_data_loader):

        # dataから学習対象データと教師ラベルデータを取り出します
        train_data, teacher_labels = data

        # 入力をtorch.autograd.Variableに変換します
        train_data, teacher_labels = Variable(train_data), Variable(
            teacher_labels)

        # 計算された勾配情報を削除します
        optimizer.zero_grad()

        # モデルに学習データを与えて予測を計算します
        outputs = model(train_data)

        # lossとwによる微分計算します
        loss = lossResult(outputs, teacher_labels)
        loss.backward()

        # 勾配を更新します
        optimizer.step()

        # 誤差を累計します
        total_loss += loss.item()

        # 2000ミニバッチずつ、進捗を表示します
        if i % 2000 == 1999:
            print('学習進捗：[%d, %d]　学習誤差（loss）: %.3f' % (epoch + 1, i + 1,
                                                      total_loss / 2000))
            total_loss = 0.0

print('学習終了')

# トータル
total = 0
# 正解カウンター
count_when_correct = 0

#
for data in test_data_loader:
    # 検証データローダーからデータを取り出した上、アンパックします
    test_data, teacher_labels = data
    # テストデータを変換した上、モデルに渡して、判定してもらいます
    results = model(Variable(test_data))
    # 予測を取り出します
    _, predicted = torch.max(results.data, 1)
    #
    total += teacher_labels.size(0)
    count_when_correct += (predicted == teacher_labels).sum()

print('count_when_correct:%d' % (count_when_correct))
print('total:%d' % (total))

print('正解率：%d / %d = %f' % (count_when_correct, total,
                            int(count_when_correct) / int(total)))
