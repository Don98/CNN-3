import random
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import axes
import numpy as np
def draw():
    a = {(100,200):10}
    a = [1,2,5]
    try:
        pos = a.index(3)
    except ValueError as e:
        pos = -1
    x=np.array([[1,2,3,4,5]])
    a = np.zeros((1,5))
    b = np.zeros((2,1))
    a[0][0] = 10
    b[0][0] = 12
    x = np.row_stack((x,a))
    x = np.column_stack((x,b))
    print(x)
    exit()
    #定义热图的横纵坐标
    xLabel = ['A','B','C','D','E']
    yLabel = ['1','2','3','4','5']
 
    #准备数据阶段，利用random生成二维数据（5*5）
    data = []
    for i in range(5):
        temp = []
        for j in range(5):
            k = random.randint(0,100)
            temp.append(k)
        data.append(temp)
    print(data)
    #作图阶段
    fig = plt.figure()
    #定义画布为1*1个划分，并在第1个位置上进行作图
    ax = fig.add_subplot(111)
    #定义横纵坐标的刻度
    ax.set_yticks(range(len(yLabel)))
    ax.set_yticklabels(yLabel)
    ax.set_xticks(range(len(xLabel)))
    ax.set_xticklabels(xLabel)
    #作图并选择热图的颜色填充风格，这里选择hot
    im = ax.imshow(data, cmap=plt.cm.hot_r)
    #增加右侧的颜色刻度条
    plt.colorbar(im)
    #增加标题
    plt.title("This is a title")
    #show
    plt.show()
 
d = draw()