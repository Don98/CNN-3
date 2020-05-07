def loadCats(self):
    file = open('classes.txt', 'r') 
    js = file.read()
    a = js[1:-2].split(", ")
    result = []
    for i in a:
        i = i.split(": ")
        dic = {}
        dic["id"] = int(i[0])
        dic["name"] = i[1][1:-1]
        result.append(dic)
    file.close()
    return result
    
def load_classes(self):
    # load class names (name -> label)
    categories = self.loadCats()
    # categories.sort(key=lambda x: x['id'])

    self.classes             = {}
    self.coco_labels         = {}
    self.coco_labels_inverse = {}
    for c in categories:
        self.coco_labels[len(self.classes)] = c['id']
        self.coco_labels_inverse[c['id']] = len(self.classes)
        self.classes[c['name']] = len(self.classes)

    # also load the reverse (label -> name)
    self.labels = {}
    for key, value in self.classes.items():
        self.labels[value] = key