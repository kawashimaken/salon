# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# 必要なモジュールをインポートします
import os

import torch
# 全てのニューラルネットワークのベースモジュール
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizer
import torchvision
import torchvision.transforms as transforms

# -----------------------------------------------------------------------------
# MacOSで、下記のエラーに遭遇する人は、下の二行が必要
# OMP: Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized.
# OMP: Hint This means that multiple copies of the OpenMP runtime have been linked into the program. That is dangerous, since it can degrade performance or cause incorrect results. The best thing to do is to ensure that only a single OpenMP runtime is linked into the process, e.g. by avoiding static linking of the OpenMP runtime in any library. As an unsafe, unsupported, undocumented workaround you can set the environment variable KMP_DUPLICATE_LIB_OK=TRUE to allow the program to continue to execute, but that may cause crashes or silently produce incorrect results. For more information, please see http://www.intel.com/software/products/support/.
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# -----------------------------------------------------------------------------
# ミニバッチのバッチサイズ
BATCH_SIZE = 4
# 最大学習回数
MAX_EPOCH = 1
# 進捗出力するバッチ数
PROGRESS_SHOW_PER_BATCH_COUNT = 1000


# -----------------------------------------------------------------------------
# ニューラルネットワークを用意します
# CNNがPyTorchのnn.Module(ニューラルネットワーククラス)を継承する
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
        # -1は自動的に変換する
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

# -----------------------------------------------------------------------------
# 学習データの準備をします
print('---------- 学習データの準備をします ----------')
# データフォーマット変更（どういうふうにデータを変更すれば良いかを指定する）の設定
# 各色チャネルの平均値
mean = (0.5, 0.5, 0.5)
# 各色チャネルの標準偏差
standard_deviations = (0.5, 0.5, 0.5)
transform = transforms.Compose([
    # データの型をTensorに変換する
    transforms.ToTensor(),
    # 色情報を標準化する
    transforms.Normalize(mean, standard_deviations)
])

# 学習データ  train=True
train_data_with_teacher_labels = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform)
train_data_loader = torch.utils.data.DataLoader(
    train_data_with_teacher_labels, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
# 検証データ train=False
test_data_with_teacher_labels = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform)
test_data_loader = torch.utils.data.DataLoader(
    test_data_with_teacher_labels, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

print('train_data_with_teacher_labels', train_data_with_teacher_labels)
# 結果：
# Dataset CIFAR10
#     Number of datapoints: 50000
#     Root location: ./data
#     Split: Train
#     StandardTransform
# Transform: Compose(
#                ToTensor()
#                Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
#            )

print('train_data_loader', train_data_loader)
# -----------------------------------------------------------------------------
# クラス名（正解教師ラベル名）
class_names = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
               'ship', 'truck')

# -----------------------------------------------------------------------------
# 学習の用意をします
# 損失関数は交差エントロピー誤差関数を使います
criterion = nn.CrossEntropyLoss()
# 最適化オプティマイザはSDGを使う、学習率lrは0.001、momentum(慣性項)を0.9に設定します
# momentumが、ここではオプションですが、転がるボールが地面の摩擦抵抗で徐々に減速していくイメージです
optimizer = optimizer.SGD(model.parameters(), lr=0.001, momentum=0.9)

print('---------- 学習開始します ----------')
# 学習開始します
for epoch in range(MAX_EPOCH):
    # 計算用誤差の初期設定
    total_loss = 0.0
    # enumerateはindexをデータを分解してくれます
    for i, data in enumerate(train_data_loader, 0):
        # dataから学習対象データと教師ラベルデータを取り出します
        train_data, teacher_labels = data

        # 計算された勾配情報を削除（リセット、クリア）します
        optimizer.zero_grad()

        # モデルに学習データを与えて予測をします
        outputs = model(train_data)

        # lossとwによる微分計算します
        loss = criterion(outputs, teacher_labels)
        # 勾配を計算します
        loss.backward()
        # 最適化のステップを一回実行します（パラメーターを更新します、たくさんのoptimizerの共通の処理）
        optimizer.step()

        # loss.item()はlossを数値に変換します、誤差を累計します
        total_loss += loss.item()

        # PROGRESS_SHOW_PER_BATCH_COUNTミニバッチずつ、進捗を表示します
        if i % PROGRESS_SHOW_PER_BATCH_COUNT == PROGRESS_SHOW_PER_BATCH_COUNT - 1:
            print(
                '学習進捗：[EPOCH:%d, %dバッチ, バッチサイズ:%d -> %d枚学習完了]　学習誤差（loss）: %.3f' % (
                    epoch + 1, i + 1, BATCH_SIZE, (i + 1) * BATCH_SIZE,
                    total_loss / PROGRESS_SHOW_PER_BATCH_COUNT))
            # 計算用誤差をリセットします
            total_loss = 0.0

print('学習完了')

# -----------------------------------------------------------------------------
# 検証
print('---------- １バッチ（画像４枚）の正確率を見てみます ----------')
# まず１バッチ（画像４枚）の正確率を見てみます
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
# ouputsも４(BATCH_SIZE)枚の写真の推論結果配列が入っています

# 一つずつ、推論結果配列の最大値（最も確信しているラベル）取り出します
print(torch.max(outputs, 1))
# 結果：
# torch.return_types.max(
# values=tensor([1.2185, 5.8557, 2.8262, 4.7874], grad_fn=<MaxBackward0>),
# indices=tensor([2, 8, 8, 8]))
# torch.max(tensor, axis)
# 　values  indices
#     ↓        ↓
#     _    predicted
_, predicted = torch.max(outputs, 1)
# 使わないものはよく、アンダーバーにします。（使い捨て）

# ここでは、axis=1なので、行ごとに最大値を取り出すという意味になります
print('_', _)
# 結果：それぞれの最大値そのものが入っています
# tensor([1.6123, 5.6203, 3.0886, 3.8317], grad_fn=<MaxBackward0>)
print('predicted', predicted)
# 結果：「最大値は何番目なのか」(index location)が入っています
# tensor([3, 9, 1, 0])
# ４(BATCH_SIZE)回実行して、class_nameから、それぞれのラベルを出します
print('予測: ', ' '.join('%5s' % class_names[predicted[j]] for j in range(BATCH_SIZE)))
#  3     9     1    0
#  ↓　　　↓　　　↓　　　↓
# cat truck   car plane

# -----------------------------------------------------------------------------
# 次は、全ての検証画像データに対しての正解率を計算します
print('---------- 全ての検証画像データに対しての正解率を計算します ----------')
# 正解カウンター
count_when_correct = 0
# 全体のデータ数（計測対象数）
total = 0
# with torch.no_grad()は、autogradが実施する自動微分のトラッキングを一時的に無効にします。
# tensor.requires_grad=True
with torch.no_grad():
    # データローダからデータバッチを取り出します
    for data in test_data_loader:
        # データと教師ラベルに分けて格納します
        test_data, teacher_labels = data
        # 学習済みモデルに渡して、推論してもらって、結果をresultsに格納します
        results = model(test_data)
        # 最大値、及び最大値の順番番号を取り出します
        _, predicted = torch.max(results.data, 1)
        # 上を参照
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
        # 4ずつ、足していきます
        # tensor predictedとtensor teacher_labelsがイコールする数を集計します
        count_when_correct += (predicted == teacher_labels).sum().item()
        # .item()は数値に変換します

print('検証画像データに対しての正解率: %d %%' % (100 * count_when_correct / total))

# -----------------------------------------------------------------------------
# 次は、全ての検証画像データに対してのクラス午後の正解率を計算します
print('---------- 全ての検証画像データに対してのクラスごとの正解率を計算します ----------')
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
#
with torch.no_grad():
    for data in test_data_loader:
        # 検証用データと正解教師ラベル四つずつ出します
        test_data, teacher_labels = data
        # 推論結果
        results = model(test_data)
        # 推論結果の結果
        _, predicted = torch.max(results, 1)
        #
        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.squeeze.html
        # 1になっているdimensionを消去します
        correct_ones = (predicted == teacher_labels).squeeze()
        # print('correct_ones',correct_ones)
        # 結果：
        # correct_ones
        # tensor([False, True, False, True])
        # それに対応して、どのクラスなのかがteacher_labels[i]に格納しています
        #
        for i in range(BATCH_SIZE):
            # 当該BATCH_SIZE個数の検証データの正解教師ラベルから、どのクラス（ラベル）なのかを決めます、例えば、
            label = teacher_labels[i]
            # print('label',label)
            # 結果：
            # label
            # tensor(3)
            # それぞれのラベルの正解の数をプラスしていきます
            class_correct[label] += correct_ones[i].item()
            class_total[label] += 1
            # print('class_correct',class_correct)
            # print('class_total',class_total)
            # 最終結果；
            # class_correct[529.0, 687.0, 296.0, 212.0, 331.0, 589.0, 707.0, 616.0, 660.0, 591.0]
            # class_total[1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0]

for i in range(10):
    # １から10のクラスの名前とその正解率を出力します
    print(' %5s クラスの正解率は: %2d %%' % (class_names[i],
                                     100 * class_correct[i] / class_total[i]))

# -----------------------------------------------------------------------------
# 終わり
# -----------------------------------------------------------------------------
