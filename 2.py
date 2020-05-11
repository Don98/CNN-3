f = open("COCO_sta.txt","r")
data = f.readlines()
part_one = [i.strip().split(" , ")[:-1] for i in data]
part_two = [i.strip().split(" , ")[-1] for i in data]
result = [" ,  || ".join([" , ".join(part_one[i]),part_two[i]])for i in range(len(data))]
f1 = open("COCO_sta1.txt","w")
for i in result:
    f1.write(str(i) + "\n")
f1.close()
f.close()