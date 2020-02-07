# -*- coding: utf-8 -*-

import numpy as np

x = np.array([[[0], [1], [2]]])
print(x.shape)
# 結果:(1, 3, 1)

after_squeeze = np.squeeze(x)

print(after_squeeze.shape)
# 結果：(3,)
print(after_squeeze)
# 結果：[0 1 2]