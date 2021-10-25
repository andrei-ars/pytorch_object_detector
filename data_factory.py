#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
First run split_dataset.py to split dataset in train|valid|test parts.
Then run this script.
"""

from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader #, random_split
import torchvision
from torchvision import datasets, models, transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor

import numpy as np
import matplotlib.pyplot as plt
import time
import os
import copy

import glob
import json
from PIL import Image

#import torchsample as ts

import settings

#IMAGE_WIDTH = 224
#IMAGE_WIDTH = 128
#RESIZE_IM = (IMAGE_WIDTH, IMAGE_WIDTH)


def get_data_transforms(image_width):

    resize_im = (image_width, image_width)

    data_transforms = {

        'train': transforms.Compose([
            #transforms.RandomResizedCrop(IMAGE_WIDTH),
            #transforms.RandomHorizontalFlip(),
            #transforms.Pad(padding=60, padding_mode='reflect'),
            #transforms.RandomRotation([0,90], expand=True),
            #transforms.RandomResizedCrop(IMAGE_WIDTH),
            #transforms.CenterCrop(IMAGE_WIDTH),

            transforms.Resize(resize_im),
            №transforms.ColorJitter(0.1, 0.1, 0, 0),

            transforms.ToTensor(),
            #ts.transforms.Rotate(20), # data augmentation: rotation 
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            #transforms.Normalize((0.5), (0.5))
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ]),

        'valid': transforms.Compose([
            transforms.Resize(resize_im),
            #transforms.CenterCrop(IMAGE_WIDTH),
            transforms.ToTensor(),
            #transforms.Normalize((0.5), (0.5))
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ]),
    }

    return data_transforms


def load_data(data_dir):
    """
    iter(ImageFolder(path)) returns a tuple like
    (<PIL.Image.Image image mode=RGB size=2550x3300 at 0x7FC4C8515828>, 0)
    """
    data_transforms = get_data_transforms()
        
    data_parts = ['train', 'valid']

    image_datasets = {p: ImageFolder(os.path.join(data_dir, p),
        data_transforms[p]) for p in data_parts}

    dataloaders = {p: DataLoader(image_datasets[p], batch_size=settings.batch_size,
        shuffle=True, num_workers=settings.num_workers) for p in data_parts}

    return dataloaders, image_datasets


def dataset_info(image_datasets):

    data_parts = list(image_datasets.keys())
    print('data_parts:', data_parts)

    dataset_sizes = {p: len(image_datasets[p]) for p in data_parts}
    for p in data_parts:
        print('{0} size: {1}'.format(p, dataset_sizes[p]))

    class_names = image_datasets['train'].classes
    print('num_classes:', len(class_names))
    print('classes:', class_names)

    return dataset_sizes, class_names


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)

#----------

def get_bbox(coords, label, img_data):

    map_label_id = {"w": 1, "t": 1, "o": 1, "a": 1, "n": 0,}
    W = img_data['width']
    H = img_data['height']
    xn = (coords['x'] / W) - 0.5
    yn = (coords['y'] / H) - 0.5
    w = coords['width'] / W
    h = coords['height'] / H
    bbox = (xn, yn, w, h, map_label_id[label])
    return bbox


def get_one_hot_class_label(coords, label, img_data, y_grid):

    S = y_grid
    map_label_id = {"w": 1, "t": 1, "o": 1, "a": 1, "n": 0,}

    W = img_data['width']
    H = img_data['height']
    x = coords['x']
    y = coords['y']
    w = coords['width']
    h = coords['height']
    #print("W={}, H={}, x={}, y={}".format(W,H,x,y))
    grid_class = int( y*S / H )
    print("grid_class:", grid_class)
    one_hot_label = [0] * S
    one_hot_label[grid_class] = 1
    #print("one_hot_label:", one_hot_label)
    return grid_class


class ObjDetDataset(Dataset):
  
  def __init__(self, images_dir, json_dir=None, data_type="train", num_colors=3,
                        image_width=224, dataset_type="obj_det", y_grid=20):
        # Initialization

    #dataset_type = "obj_det"
    #dataset_type = "grid_class"

    #transform = data_transforms['train']
    transform = get_data_transforms(image_width)['train']

    self.data = []

    json_dir = json_dir if json_dir else images_dir

    for file_path in glob.glob(os.path.join(json_dir, "*.json")):
        print("{}".format(file_path))
        with open(file_path) as fp:

            json_data = json.load(fp)[0]
            img_path = os.path.join(images_dir, json_data.get('image'))
            img = Image.open(img_path)
            img_width, img_height = img.size

            if num_colors == 3:
                rgbimg = Image.new("RGB", img.size)
                rgbimg.paste(img)
                img = rgbimg
            
            img_tensor = transform(img)
            #print("img_tensor.shape:", img_tensor.shape)

            img_data = {}
            img_data['file_name'] = img_path
            img_data['image_id'] = os.path.basename(img_path)
            img_data['height'] = img_height
            img_data['width'] = img_width

            annotations = json_data.get("annotations")
            if annotations:
                for ann in annotations:
                    #label = ann["label"]
                    #coord = ann["coordinates"]
                    #bbox = (coord['x'], coord['y'], coord['width'], coord['height'], map_label_id[label])
                    #bbox = torch.tensor(bbox)
                    if dataset_type == "obj_det":
                        bbox = get_bbox(ann["coordinates"], ann["label"], img_data)
                        label_tensor = torch.tensor(bbox)
                    elif dataset_type == "grid_class":
                        class_label = get_one_hot_class_label(
                            ann["coordinates"], ann["label"], img_data, y_grid=y_grid)
                        label_tensor = torch.tensor(class_label)
                    self.data.append( (img_tensor, label_tensor) )
                    #print((img, bbox))
            else:
                #bbox = (coord['x'], coord['y'], coord['width'], coord['height'], map_label_id[label])
                #bbox = torch.tensor(bbox)
                #bbox = get_bbox((), ann["label"], img_data)
                if dataset_type == "obj_det":
                    label_tensor = torch.tensor( (0.5, 0.5, 0.5, 0.5, 0) )
                elif dataset_type == "grid_class":
                    label_tensor = torch.tensor(0)
                self.data.append( (img_tensor, label_tensor) )
                #print((img, bbox))

        #self.labels = labels
        #self.list_IDs = list_IDs

  def __len__(self):
        #Denotes the total number of samples
        return len(self.data)

  def __getitem__(self, index):
        #Generates one sample of data
        # Select sample
        #ID = self.list_IDs[index]
        # Load data and get label
        #X = torch.load('data/' + ID + '.pt')
        #y = self.labels[ID]
        #return X, y
        return self.data[index]


if __name__ == '__main__':

    """
    data_dir = settings.data_dir #'/w/WORK/ineru/06_scales/_dataset/splited/'
    data_dir = "/data/5_patexia/32_object_detection/dataset/"
    dataloaders, image_datasets = load_data(data_dir)
    dataset_sizes, class_names = dataset_info(image_datasets)

    # Get a batch of training data
    for inputs, classes in dataloaders['valid']:
        print(len(inputs))
        print(classes)
    
        out = torchvision.utils.make_grid(inputs)
        imshow(out, title=[class_names[x] for x in classes])
        plt.pause(2)  # pause a bit so that plots are updated
    """

    #train_dataset = ObjDetDataset(
    #    images_dir="/data/tools/dataset/train/", 
    #    json_dir="/data/tools/dataset/json_annotations/train/"
    #    )

    data_path = settings.data_path
    dataset = ObjDetDataset(
        #images_dir="../dataset/train/", 
        #json_dir="../dataset/json_annotations/train/",
        images_dir=os.path.join(settings.data_path, "valid"), 
        json_dir=os.path.join(settings.data_path, "json_annotations/valid"),
        data_type="valid", 
        num_colors=settings.num_colors,
        image_width=settings.image_width,
        dataset_type = "grid_class",
        y_grid=20
    )    

    print("len(dataset):", len(dataset))
    print(dataset[0])
    print(dataset[1])
