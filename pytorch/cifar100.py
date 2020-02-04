import torch
import torchvision
import torchvision.transforms as transforms
#
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optimizer
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# ミニバッチのサイズ
BATCH_SIZE = 10
# 最大学習回数
MAX_EPOCH = 4


# 学習データ
train_data_with_teacher_labels = torchvision.datasets.CIFAR100(
    root='./data', train=True, download=True, transform=transform)
train_data_loader = torch.utils.data.DataLoader(train_data_with_teacher_labels,
                                                batch_size=BATCH_SIZE,
                                                shuffle=True,
                                                num_workers=2)
# 検証データ
test_data_with_teacher_labels = torchvision.datasets.CIFAR100(
    root='./data', train=False, download=True, transform=transform)
test_data_loader = torch.utils.data.DataLoader(test_data_with_teacher_labels,
                                               batch_size=BATCH_SIZE,
                                               shuffle=False,
                                               num_workers=2)

cifar100_labels = (
   'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
    'worm'
)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        # 入力チャネル、出力チャネル、カーネル（フィルタ）サイズ、ストライド、パディング、dilation、グループ、バイアスあり、ゼロパディング
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.layer1 = nn.Linear(16 * 5 * 5, 240)
        self.layer2 = nn.Linear(240, 240)
        self.layer3 = nn.Linear(240, 100)

    def forward(self, input_data):
        input_data = self.pool(F.relu(self.conv1(input_data)))
        input_data = self.pool(F.relu(self.conv2(input_data)))
        input_data = input_data.view(-1, 16 * 5 * 5)
        input_data = F.relu(self.layer1(input_data))
        input_data = F.relu(self.layer2(input_data))
        input_data = self.layer3(input_data)
        return input_data


model = CNN()

criterion = nn.CrossEntropyLoss()
optimizer = optimizer.SGD(model.parameters(), lr=0.001, momentum=0.9)



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
            print('学習進捗：[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, total_loss / 2000))
            total_loss = 0.0

print('学習完了')

count_when_correct = 0
total = 0
with torch.no_grad():
    for data in test_data_loader:
        test_data, teacher_labels = data
        results = model(test_data)
        _, predicted = torch.max(results.data, 1)
        # teacher_labels.size(0)=4 一セットのtest_dataは４枚の画像
        total += teacher_labels.size(0)

        count_when_correct += (predicted == teacher_labels).sum().item()

print('10000 検証画像に対しての正解率: %d %%' % (100 * count_when_correct / total))

class_correct = list(0. for i in range(100))
class_total = list(0. for i in range(100))
#
with torch.no_grad():
    for data in test_data_loader:
        #
        test_data, teacher_labels = data
        # print('test_data.shape',test_data.shape)
        # test_dataのshapeはtorch.Size([4, 3, 32, 32])
        #
        results = model(test_data)
        # print('results.shape',results.shape)
        # resultsのshapeはtorch.Size([4, 100])
        # 最大値を取り出す
        _, predicted = torch.max(results, 1)
        # predictedのshapleはtorch.Size([4])
        # 予測が教師ラベルと一致する場合、次元を圧縮する
        # trueの個数（何個正解したか）
        correctly_preidicted = (predicted == teacher_labels).squeeze()
        #
        for i in range(BATCH_SIZE):
            label = teacher_labels[i]
            # クラスごとに、正確に推論した数の累計
            class_correct[label] += correctly_preidicted[i].item()
            # 正解教師ラベル数の累計
            class_total[label] += 1

for i in range(100):
    print(' %5s クラスの正解率は: %2d %%' %
          (cifar100_labels[i], 100 * class_correct[i] / class_total[i]))
