import os

files = os.listdir("")
classes = len(files)/4
f = open("","w")
clas = {}
num = 0
for i in file[::4]:
    clas[num] = i.split("_")[0]
f.write(str(clas))