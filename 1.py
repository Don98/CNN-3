import json
f = open("./instances_val2017.json","r")
data = json.load(f)
print(data.keys())
print(data['images'][0])
print(data['annotations'][0])
f.close()