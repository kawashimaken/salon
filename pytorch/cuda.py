import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# GPUがある場合は、cuda:0と表示する、ない場合は、cpuと表示する
print(device)
