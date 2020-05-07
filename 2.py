file = open('classes.txt', 'r') 
js = file.read()
a = js[1:-2].split(", ")
dic = {}
for i in a:
    i = i.split(": ")
    dic[int(i[0])] = i[1][1:-1]
file.close()
print(dic)