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
                                    transform=transforms.Compose([Normalizer(), Augmenter(), Resizer([460,640])]),part = 1)
        dataset_val = CocoDataset(parser.coco_path, set_name='val2017',
                                  transform=transforms.Compose([Normalizer(), Resizer([460,640])]),part = 1)
                                  # transform=transforms.Compose([Normalizer(), Resizer([350,500])]),part = 1)
    print(len(dataset_train))
    for i in dataset_train:
        # print(i["img"])
        pass

if __name__ == "__main__":
    main()