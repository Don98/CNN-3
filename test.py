import argparse
import collections

import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms

from CNN3 import model
from CNN3.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
    Normalizer
from torch.utils.data import DataLoader

from CNN3 import coco_eval
from CNN3 import csv_eval

assert torch.__version__.split('.')[0] == '1'

from PIL import Image

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a cnn3 network.')

    parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.')
    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--csv_train', help='Path to file containing training annotations (see readme)')
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')

    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=101)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=100)

    parser = parser.parse_args(args)

    # Create the data loaders
    
    if parser.dataset == 'coco':

        if parser.coco_path is None:
            raise ValueError('Must provide --coco_path when training on COCO,')

        dataset_train = CocoDataset(parser.coco_path, set_name='train2017',
                                    transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]),part = 10)
        dataset_val = CocoDataset(parser.coco_path, set_name='val2017',
                                  transform=transforms.Compose([Normalizer(), Resizer()]),part = 10)
    scale_list = {}
    import matplotlib.pyplot as plt
    f = open("h_w_scale_hw.txt","w")
    for i in dataset_train:
    	f.write(str(i["img"].size()) +  " " + str(i["scale"]) + str(i["img"].size()[0] * i["img"].size()[1] * i["img"].size()[2] / 3) + "\n")
	    # print(i.keys())
	    # exit()
    #     if(i["scale"] in scale_list):
    #         scale_list[i["scale"]] += 1
    #     else:
    #         scale_list[i["scale"]] = 0
    # X = sorted(scale_list.keys())
    # Y = []
    # result = {}
    # for i in X:
    #     Y.append(scale_list[i])
    #     result[i] = Y[-1]
    # f = open("scale_info.txt","w")
    # f.write(str(result))
    f.close()
    # plt.bar(X,Y)
    # plt.show()

if __name__ == "__main__":
    main()