import numpy as np

import torch
my_tensor = torch.tensor([[1, 2], [3, 4]])
# CUDAがある場合
# my_tensor = torch.tensor([[1, 2], [3, 4]], device="cuda:0")

my_tensor = torch.tensor([[1, 2], [3, 4]], dtype=torch.float64)

my_tensor = torch.arange(0, 100)

my_tensor = torch.randn(10, 10)
print(my_tensor.size())

np_array = my_tensor.numpy()

# PyTorchのTensorからNumPyの配列に変換ができる
print(np_array)
