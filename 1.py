import random
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import axes
import numpy as np
def draw():
    f = open("h_w_scale_hw1.txt","r")
    data = f.readlines()
    data = [(int(i[12:i.index(",")]),int(i[i.index(",")+2:i.index(",",i.index(",")+1)])) for i in data[::2]]
    f.close()
    scale_list = {}
    pos_list = np.array([[0]])
    from matplotlib import pyplot as plt
    from matplotlib import cm
    from matplotlib import axes
    X = [0]
    Y = [0]
    f = open("scale_h_w.txt","w")
    for i in data:
        part = (i[0],i[1])
        if(part in scale_list):
            scale_list[part] += 1
            pos_list[pos_x][pos_y] += 1
        else:
            scale_list[part] = 1
            if part[0] in X:
                pos_x = X.index(part[0])
            else:
                pos_x = len(X)
                for j in range(len(X)):
                    if(X[j] < part[0]):
                        pos_x = j
                        break
                X.insert(pos_x,part[0])
                to_insert = np.zeros((1,len(Y)))
                pos_list = np.insert(pos_list,pos_x,values=to_insert,axis=0)
            if part[1] in Y:
                pos_y = Y.index(part[1])
            else:
                pos_y = len(Y)
                for j in range(len(Y)):
                    if(Y[j] > part[1]):
                        pos_y = j
                        break
                Y.insert(pos_y,part[1])
                to_insert = np.zeros((1,len(X)))
                pos_list = np.insert(pos_list,pos_y,values=to_insert,axis=1)
            pos_list[pos_x][pos_y] += 1
    X.pop(-1)
    Y.pop(0)
    pos_list = np.delete(pos_list,-1,axis=0)
    pos_list = np.delete(pos_list,0,axis=1)
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.set_yticks(range(len(Y)))
    # ax.set_yticklabels(Y)
    # ax.set_xticks(range(len(X)))
    # ax.set_xticklabels(X)
    
    # im = ax.imshow(pos_list, cmap=plt.cm.hot_r)
    # plt.colorbar(im)
    # plt.title("This is the scale of coco")
    # plt.show()
    import pandas as pd
    import seaborn as sns
    print(X)
    print(Y)
    df = pd.DataFrame(pos_list,index = X ,columns = Y)
    print(df.head())
    
    
    sns.set()
    ax = sns.heatmap(df,annot=True, fmt='d', linewidths=.5, cmap='YlGnBu')
    # ax = sns.heatmap(df,annot=True, fmt='d', linewidths=.5, cmap='RdBu')
    plt.title("This is the scale of coco train")
    plt.show()
    # df.to_csv("pan.csv")
    
    # print(len(X),len(Y))
    # print(pos_list.shape)
    # exit()
    
    # X.insert(0,0)
    # pos_list = np.insert(pos_list,0,values = Y,axis=0)
    # pos_list = np.insert(pos_list,0,values = X,axis=1)
    # np.savetxt("temp.csv", pos_list, delimiter=",")
 
d = draw()