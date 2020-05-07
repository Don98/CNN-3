import re
f= open("000001.xml","r")
data = f.read()
objects = re.compile("<object>([\w\W]+?)</object>").findall(data)
# print(objects[0])
# print(objects[1])
result = []
for i in objects:
    name = re.compile("<name>([\w\W]+?)</name>").findall(i)[0].strip()
    bndbox = re.compile("<bndbox>([\w\W]+?)</bndbox>").findall(i)[0].strip()
    print(name)
    nums = re.compile("<[\w\W]+?>([\w\W]+?)</[\w\W]+?>").findall(bndbox)
    nums.append(name)
    result.append(nums)
f.close()
# return result
print(result)