import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt


def cluster_model(x, model):
    if (model == "dbscan"):
        dbscan = DBSCAN(eps=1, min_samples=5, metric='euclidean').fit(x)
        dbscan.fit(x)
        return dbscan.labels_
    if (model == "kmeans"):
        kms = KMeans(n_clusters=2)
        return kms.fit_predict(x)


if __name__ == "__main__":
    f = open("file/COCO/scale_h_w_or.txt", "r")
    data = f.readlines()
    data = [[int(i[1:i.index(",")]), int(i[i.index(",") + 2:-2])] for i in data[::2]]
    # print(data)
    f.close()
    x = np.array(data)

    y = cluster_model(x, "kmeans")
    # y = cluster_model(x,"dbscan")
    y = np.array(y)
    fig = plt.figure()
    ax = plt.subplot()
    ax.scatter(x[y == 0][:, 0], x[y == 0][:, 1], alpha=0.5)
    ax.scatter(x[y == 1][:, 0], x[y == 1][:, 1], c='green', alpha=0.5)
    plt.show()