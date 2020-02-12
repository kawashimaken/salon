# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------

import torch

print(torch.__version__)

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torch.autograd import Variable
import torch.nn as nn

import torch.optim as optimizer

# -----------------------------------------------------------------------------
# ミニバッチのバッチサイズ
BATCH_SIZE = 4
# 最大学習回数
MAX_EPOCH = 2
# 進捗出力するバッチ数
PROGRESS_SHOW_PER_BATCH_COUNT=1000


# -----------------------------------------------------------------------------
# 　マルチレイヤーパーセプトロンクラスの定義
class MLP(nn.Module):
    def __init__(self):
        '''
        層ごとに定義する、例えば、活性化関数などは、次forward()で定義する
        '''
        super().__init__()
        # 入力層
        self.layer1 = nn.Linear(28 * 28, 100)
        # 中間層（隠れ層）
        self.layer2 = nn.Linear(100, 50)
        # 出力層
        self.layer3 = nn.Linear(50, 10)

    def forward(self, input_data):
        '''
        ネットワークの（順伝播）の定義（つなげる）
        '''
        # input_dataをフォーマット変換します
        # -1は自動的に変換する
        input_data = input_data.view(-1, 28 * 28)
        # 前の層からきたinput_dataをlayer1に渡します
        input_data = self.layer1(input_data)
        # 前の層からきたinput_dataをlayer2に渡します
        input_data = self.layer2(input_data)
        # 前の層からきたinput_dataをlayer3に渡します
        input_data = self.layer3(input_data)
        return input_data


# 学習用モデルのインスタンスを生成します
model = MLP()

# -----------------------------------------------------------------------------
# 学習データの準備をします
#
print('---------- 学習のデータの準備 ----------')
data_folder = '~/data'
transform = transforms.Compose([
    # データの型をTensorに変換する
    transforms.ToTensor()
])

# 学習データ
train_data_with_labels = FashionMNIST(
    data_folder, train=True, download=True, transform=transform)

train_data_loader = DataLoader(
    train_data_with_labels, batch_size=BATCH_SIZE, shuffle=True)

# 検証データ
test_data_with_labels = FashionMNIST(
    data_folder, train=False, download=True, transform=transform)
test_data_loader = DataLoader(
    test_data_with_labels, batch_size=BATCH_SIZE, shuffle=True)

# -----------------------------------------------------------------------------
# 学習の用意をします
# 損失関数は交差エントロピー誤差関数を使います
lossResult = nn.CrossEntropyLoss()
# SGD
optimizer = optimizer.SGD(model.parameters(), lr=0.01)

print('---------- 学習開始します ----------')
# 学習開始します
for epoch in range(MAX_EPOCH):
    # 誤差の初期設定
    total_loss = 0.0
    # enumerateはindexをデータを分解してくれます
    for i, data in enumerate(train_data_loader):

        # dataから学習対象データと教師ラベルデータのバッチを取り出します
        train_data, teacher_labels = data

        # 入力をtorch.autograd.Variableに変換します
        train_data, teacher_labels = Variable(train_data), Variable(
            teacher_labels)

        # 計算された勾配情報を削除（リセット、クリア）します
        optimizer.zero_grad()

        # モデルに学習データを与えて予測をします
        outputs = model(train_data)

        # lossとwによる微分計算します
        loss = lossResult(outputs, teacher_labels)
        # 勾配を計算します
        loss.backward()

        # 最適化のステップを一回実行します（パラメーターを更新します、たくさんのoptimizerの共通の処理）
        optimizer.step()

        # loss.item()はlossを数値に変換します、誤差を累計します
        total_loss += loss.item()

        # PROGRESS_SHOW_PER_BATCH_COUNTミニバッチずつ、進捗を表示します
        if i % PROGRESS_SHOW_PER_BATCH_COUNT == PROGRESS_SHOW_PER_BATCH_COUNT-1:
            print('i=',i)
            print(
                '学習進捗：[EPOCH:%d, %dバッチx%d -> %d枚学習完了]　学習誤差（loss）: %.3f' % (epoch + 1, i + 1, BATCH_SIZE, (i + 1) * BATCH_SIZE,
                                                                     total_loss / PROGRESS_SHOW_PER_BATCH_COUNT))
            # 計算用誤差をリセットします
            total_loss = 0.0

print('学習終了')

# -----------------------------------------------------------------------------
# 検証：全ての検証画像データに対しての正解率を計算します
print('---------- 全ての検証画像データに対しての正解率を計算します ----------')
# 全体のデータ数（計測対象数）
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
    print(torch.max(results, 1))
    # 結果：
    # torch.return_types.max(
    # values=tensor([1.2185, 5.8557, 2.8262, 4.7874], grad_fn=<MaxBackward0>),
    # indices=tensor([2, 8, 8, 8]))
    # torch.max(tensor, axis)
    # 　values  indices
    #     ↓        ↓
    #     _    predicted
    _, predicted = torch.max(results.data, 1)
    # 一つずつ、推論結果配列の最大値（最も確信しているラベル）取り出します
    # 使わないものはよく、アンダーバーにします。（使い捨て）
    # ここでは、axis=1なので、行ごとに最大値を取り出すという意味になります
    print('_', _)
    # 結果：それぞれの最大値そのものが入っています
    # tensor([1.6123, 5.6203, 3.0886, 3.8317], grad_fn=<MaxBackward0>)
    print('predicted', predicted)
    # 結果：「最大値は何番目なのか」(index location)が入っています
    # tensor([3, 9, 1, 0])
    #
    # print('teacher_labels',teacher_labels)
    # 結果：
    # teacher_labels
    # tensor([3, 5, 3, 8])
    # teacher_labels
    # tensor([3, 5, 1, 7])
    # ...
    # ...
    #
    # print('teacher_labels.size(0)',teacher_labels.size(0))
    # teacher_labels.size(0) 4
    total += teacher_labels.size(0)
    count_when_correct += (predicted == teacher_labels).sum()

print('count_when_correct:%d' % (count_when_correct))
print('total:%d' % (total))

print('正解率：%d / %d = %f' % (count_when_correct, total,
                            int(count_when_correct) / int(total)))
