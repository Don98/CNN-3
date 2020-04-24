### RetinaNet 代码总结

RetuinaNet有两种引入数据集的方式，分别是COCO个CSV。

```
python train.py --dataset coco --coco_path ../coco --depth 50
```

代码从train.py开始解读。

![image-20200423110347972](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20200423110347972.png)

首先使用argparse来读取输入的参数。

#### 判断数据集

```python
if parser.dataset == 'coco':

	if parser.coco_path is None:
            raise ValueError('Must provide --coco_path when training on COCO,')
	dataset_train = CocoDataset(parser.coco_path, set_name='train2017',
		transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
    dataset_val = CocoDataset(parser.coco_path, set_name='val2017',
        transform=transforms.Compose([Normalizer(), Resizer()]))
```

新建实例CocoDataset，这是一个自定义于retinanet.dataloader的类。

CocoDataset的第一个参数是数据集的位置，第二个是名字，然后第三个是一个torch的transforms。这一个transforms，引入了Normalizer()、Augmenter()、Resizer()，这三个类(都是自定义的)，分别对于数据集进行正则化、转化成tensor和resize。



#### 生成dataloader

    sampler = AspectRatioBasedSampler(dataset_train, batch_size=2, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=3,collate_fn=collater, 	batch_sampler=sampler)
    if dataset_val is not None:
    	sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1,
    		drop_last=False)
       	dataloader_val = DataLoader(dataset_val, num_workers=3, collate_fn=collater, batch_sampler=sampler_val)
首先借用retinanet.dataloader内的AspectRatioBasedSampler类生成一个sampler，然后加载到dataloader_train中，当然也生成验证集的dataloader。



#### 选用合适的ResNet模型

    # Create the model
    if parser.depth == 18:
        retinanet = model.resnet18(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 34:
        retinanet = model.resnet34(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 50:
        retinanet = model.resnet50(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 101:
        retinanet = model.resnet101(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 152:
        retinanet = model.resnet152(num_classes=dataset_train.num_classes(), pretrained=True)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')


#### 多个GPU

如果有多个GPU的话就多个GPU一起用

    if torch.cuda.is_available():
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet = torch.nn.DataParallel(retinanet)


#### 准备训练

```
retinanet.training = True
```

#### 设置优化器

    optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
设置Adam优化器和设置学习率的调整可以基于一些验证测试进行动态的下降，设置在3个epoch之后没有提升才修改学习率，verbose=True表示在更新的时候进行输出。

#### 设置loss_hist

```
loss_hist = collections.deque(maxlen=500)
```

用来记录los。估计后面也可以用来画图。

#### 冻结BN

    retinanet.train()
    retinanet.module.freeze_bn()
#### 开始训练

```
for iter_num, data in enumerate(dataloader_train):
	try:
    	optimizer.zero_grad()
		if torch.cuda.is_available():
        	classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot']])
        else:
        	classification_loss, regression_loss = retinanet([data['img'].float(),
        	data['annot']])
            classification_loss = classification_loss.mean()
            regression_loss = regression_loss.mean()
            loss = classification_loss + regression_loss

            if bool(loss == 0):
				continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)
			optimizer.step()
			loss_hist.append(float(loss))
			epoch_loss.append(float(loss))
			print('Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist)))

            del classification_loss
            del regression_loss
    except Exception as e:
    	print(e)
        continue
```

首先是用zero_grad把梯度清零，然后分别计算一个分类误差和一个回归误差，计算出均值之后加起来，然后对loss进行反向传播，同时记得梯度裁剪，防止梯度爆炸和消失。

        if parser.dataset == 'coco':
            print('Evaluating dataset')
            coco_eval.evaluate_coco(dataset_val, retinanet)
evaluate_coco是自己写的文件，本质上是在统计在这个数据集上的效果。

#### 保存模型

每训练一个ecpoh都保存一次

```
torch.save(retinanet.module, '{}_retinanet_{}.pt'.format(parser.dataset, epoch_num))
```

```
torch.save(retinanet, 'model_final.pt')#保存最后的模型
```

