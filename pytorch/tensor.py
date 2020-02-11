# -*- coding: utf-8 -*-

import torch

# -----------------------------------------------------------------------------
# tensorの作成
# torch.tensorは多次元行列　numpyのように
# 一つのデータタイプを持っている
my_tensor = torch.tensor([[1, 2], [3, 4]])
# CUDAがある場合
# my_tensor = torch.tensor([[1, 2], [3, 4]], device="cuda:0")

# -----------------------------------------------------------------------------
# tensorのデータタイプの指定　dtype=XXXXX
my_tensor = torch.tensor([[5, 6], [7, 8]], dtype=torch.float64)
# torch.float64 64ビット倍精度浮動小数点型（符号部1ビット、指数部11ビット、仮数部52ビット）
# torch.float32 32ビット単精度浮動小数点型（符号部1ビット、指数部8ビット、仮数部23ビット）
# https://pytorch.org/docs/stable/tensors.html

# -----------------------------------------------------------------------------
# tensorの作成 要素がゼロ
my_tensor = torch.zeros([2, 2], dtype=torch.int32)
print(my_tensor)
# -----------------------------------------------------------------------------
# tensorの作成 arange
my_tensor = torch.arange(0, 100)

# -----------------------------------------------------------------------------
# tensorの作成 randn
# 10x10の2次元tensorを作成
my_tensor = torch.randn(10, 10)
print(my_tensor.size())
# 結果：
# torch.Size([10, 10])
# tensorの内容省略
print(my_tensor)

my_tensor = torch.rand(5, 3)
print(my_tensor.size())
# 結果：
# torch.Size([5, 3])
# [[0.29979497 0.51767975 0.5463143 ]
#  [0.23490947 0.98131263 0.61104506]
#  [0.70910174 0.5823815  0.62210214]
#  [0.11653227 0.48275697 0.3188144 ]
#  [0.3246895  0.42060983 0.8447304 ]]
print(my_tensor[1][1])

# -----------------------------------------------------------------------------
# tensorからnumpay 配列へ変換
np_array = my_tensor.numpy()

# PyTorchのTensorからNumPyの配列に変換ができる
print(np_array)

my_tensor = torch.tensor([[1., -1.], [1., 1.]])
print(my_tensor)
# アンダーバーがついていない関数が、tensorを変えない
print(torch.FloatTensor.abs(my_tensor))
print(my_tensor)
# アンダーバーがついている関数が、tensorを変えてしまう
print(torch.FloatTensor.abs_(my_tensor))
print(my_tensor)
