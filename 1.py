with open("h_w_scale_hw.txt","r") as f:
    data = f.readlines()
result = []
f = open("h_w_scale_hw1.txt","w")
for i in data:
    # print(i)
    # i = "torch.Size([640, 832, 3]) 1.2666666666666666532480.0"
    a = i.find("[")
    b = i.find("]")
    num = i[a+1:b].replace(" ","").split(",")
    num = len(str(int(num[0]) * int(num[1])))
    f.write(i[:- num - 3] + " " + i[-num-3:] + "\n")
f.close()