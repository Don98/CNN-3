import numpy as np

def get_center(path):
    f = open(path,"r")
    data = f.readlines()
    data = [[int(i[1:i.index(",")]), int(i[i.index(",") + 2:-2])] for i in data]
    f.close()
    return data

if __name__ == "__main__":
    f = open("file/COCO/scale_h_w_or.txt","r")
    center_path = "center_kmeans.txt"
    data = f.readlines()
    f.close()
    data = [[int(i[1:i.index(",")]), int(i[i.index(",") + 2:-2])] for i in data[::2]]
    x = np.array(data)
    ccenter = get_center("center_kmeans.txt")
    print(data)