import plotly.graph_objects as go
import numpy as np
from sklearn import decomposition
from sklearn import datasets

def getcolor(c):
    if c==0:
        return 'red'
    elif c==1:
        return 'orange'
    elif c==2:
        return 'yellow'
    elif c==3:
        return 'greenyellow'
    elif c==4:
        return 'green'
    elif c==5:
        return 'cyan'
    elif c==6:
        return 'blue'
    elif c==7:
        return 'navy'
    elif c==8:
        return 'purple'
    else:
        return 'black'
        
# 手書き数字のデータをロードし、変数digitsに格納
digits = datasets.load_digits()

# 特徴量のセットを変数Xに、ターゲットを変数yに格納
all_features = digits.data
teacher_labels = digits.target

"""（0-9）数字データの色を指定する関数です。"""

# 主成分分析を行って、3次元へと次元を減らします
pca = decomposition.PCA(n_components=3)

# 主成分分析により、64次元のall_featuresを3次元のthree_featuresに変換
three_features = pca.fit_transform(all_features)

# Helix equation
t = np.linspace(0, 10, 50)
x, y, z = three_features[:,0], three_features[:,1], three_features[:,2]

fig = go.Figure(data=[go.Scatter3d(
    x=x, 
    y=y, 
    z=z,
    mode='markers',
    marker=dict(
        size=5,
        color=list(map(getcolor, teacher_labels)),           # set color to an array/list of desired values
        
        opacity=0.8
    ))])
fig.show()