PyTorchを使ったプログラムです。
```
pip install torch
pip install torchvision
```

実行するには、下記のコマンドを実行してください。
```
python xxxx.py
```

## [mnist.py](mnist.py)

手書き数字認識　http://yann.lecun.com/exdb/mnist/

## [fashion_mnist.py](fashion_mnist.py)

10種類の小さい衣類等の画像分類　https://github.com/zalandoresearch/fashion-mnist

## [kmnist.py](kmnist.py)

崩字の認識　https://github.com/rois-codh/kmnist

## [emnist.py](emnist.py)

MNISTの拡張　https://www.westernsydney.edu.au/bens/home/reproducible_research/emnist

データダウンロード、時間がかかる

## [qmnist.py](qmnist.py)

MNISTデータの再構築　

https://www.nist.gov/srd/nist-special-database-19　
からの再構築
https://github.com/facebookresearch/qmnist

## [cnn_cifar10.py](cnn_cifar10.py)

CIFAR-10の画像分類（10種類）　https://www.cs.toronto.edu/~kriz/cifar.html

## [cnn_cifar100.py](cnn_cifar100.py)

CIFAR-100の画像分類（100種類）　https://www.cs.toronto.edu/~kriz/cifar.html

学習、やや時間がかかる

## [vgg16.py](vgg16.py)

VGG16の学習済みモデルを使って、犬の（写真を使って犬の）品種を認識するプログラム

## [vgg19.py](vgg19.py)

今、VGG19のモデルがダウンロードできない状況が発生しています（2020/01/12）。https://github.com/pytorch/vision/issues/1876
復旧したようです（2020/01/13）。

VGG16の学習済みモデルを使って、犬の（写真を使って犬の）品種を認識するプログラム

