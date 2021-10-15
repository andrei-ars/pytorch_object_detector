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
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision import datasets, models, transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor

import math
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys
import copy
import progressbar
from nn_models import get_torchvision_model, get_resnet18_classifier
from nn_models import CNN_Net
import data_factory
from data_factory import ObjDetDataset
from loss_function import bboxes_loss
SHOW_BAR = False

import settings
model_name = settings.model_name

if model_name == "resnet":
    num_colors = 3
    image_width = 224
elif model_name == "custom":
    num_colors = 1
    image_width = 128

data_parts = ['train', 'valid']
#batch_size = settings.batch_size
batch_size = 4

if True:

    train_dataset = ObjDetDataset(
        images_dir="../dataset/train/", 
        json_dir="../dataset/json_annotations/train/",
        #json_dir="/data/5_patexia/32_object_detection/dataset/_small_dataset/train/",
        data_type="train", 
        num_colors=num_colors,
        image_width=image_width,
    )
    valid_dataset = ObjDetDataset(
        images_dir="../dataset/valid/", 
        json_dir="../dataset/json_annotations/valid/",
        #json_dir="/data/5_patexia/32_object_detection/dataset/_small_dataset/valid/",
        data_type="valid", 
        num_colors=num_colors,
        image_width=image_width,
    )
    dataset = {'train': train_dataset, 'valid': valid_dataset}
    dataset_sizes = {'train': len(dataset['train']), 'valid': len(dataset['valid']) }

    num_batch = dict()
    num_batch['train'] = math.ceil(dataset_sizes['train'] / batch_size)
    num_batch['valid'] = math.ceil(dataset_sizes['valid'] / batch_size)
    print('train_num_batch:', num_batch['train'])
    print('valid_num_batch:', num_batch['valid'])

    dataloaders = {
        'train': DataLoader(dataset['train'], batch_size=batch_size, shuffle=True),
        'valid': DataLoader(dataset['valid'], batch_size=batch_size, shuffle=True)
    }


#print(data_parts)
#print('train size:', dataset_sizes['train'])
#print('valid size:', dataset_sizes['valid'])
#print('classes:', class_names)
#print('class_to_idx:', dataset.class_to_idx)

#for i, (x, y) in enumerate(dataloaders['valid']):
#    print(x) # image
#    print(i, y) # image label


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    history = dict()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        history[epoch] = {}

        # Each epoch has a training and validation phase
        for phase in data_parts:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            if SHOW_BAR: 
                bar = progressbar.ProgressBar(maxval=num_batch[phase]).start()

            # Iterate over data.
            for i_batch, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                if SHOW_BAR: 
                    bar.update(i_batch)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    print("outputs:", outputs)
                    print("labels:", labels) # labels: tensor([1, 2, 2, 2])
                    print("loss:", loss) # tensor(1.1003, grad_fn=<NllLossBackward>)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                #print('preds: ', preds)
                #print('labels:', labels.data)
                #print('match: ', int(torch.sum(preds == labels.data)))

                running_loss += loss.item() * inputs.size(0)

            if SHOW_BAR: 
                bar.finish()

            epoch_loss = running_loss / dataset_sizes[phase]

            print('{} Loss: {:.4f}'.format(phase, epoch_loss))
            history[epoch][phase] = {'loss': epoch_loss}

            # deep copy the model
            #if phase == 'valid' and epoch_acc > best_acc:
            #    best_acc = epoch_acc
            #    best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    #print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    #model.load_state_dict(best_model_wts)

    print("\nEpoch Train Valid")
    for epoch in range(num_epochs):
        train_loss = history[epoch]['train']['loss']
        valid_loss = history[epoch]['valid']['loss']
        print('{}: {:.4f} {:.4f}'.format(epoch, train_loss, valid_loss))

    return model


if __name__ == "__main__":

    if model_name == "resnet":
        #model = get_resnet18_classifier(output_size=5, pretrained=False)
        model = get_torchvision_model(output_size=5, pretrained=True)
    elif model_name == "custom":
        model = CNN_Net(output_size=5, num_input_channels=1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    #criterion = nn.CrossEntropyLoss()
    criterion = bboxes_loss

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    #optimizer_ft = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=3, gamma=0.1)

    model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler,
        num_epochs=8)

    # save model
    torch.save(model.state_dict(), "model_state.pt")
    torch.save(model, "model_full.pt")


# Results:

# 19: 0.0474 0.0354

# resnet18:
# 18: 0.0098 0.0477
# 19: 0.0099 0.0474
# --
# 19: 0.0097 0.0491

# custom:
#2: 0.0143 0.0057
#3: 0.0131 0.0053

"""


18: 0.0335 0.1952
19: 0.0407 0.1900

Epoch Train Valid
0: 0.1269 0.1348
1: 0.0517 0.0342
2: 0.0310 0.0498
3: 0.0301 0.0281
4: 0.0198 0.0147


lr=0.003
Epoch Train Valid
0: 0.0974 0.0842
1: 0.0533 0.0288
2: 0.0357 0.0469
3: 0.0199 0.0417
4: 0.0156 0.0213
5: 0.0128 0.0191
6: 0.0076 0.0111
7: 0.0069 0.0120


Epoch Train Valid
0: 0.1968 0.0760
1: 0.0462 0.0407
2: 0.0369 0.1018
3: 0.0233 0.0427
4: 0.0203 0.0203
5: 0.0125 0.0291
6: 0.0076 0.0107
7: 0.0055 0.0109


Epoch Train Valid
0: 0.0711 0.0152
1: 0.0160 0.0091
2: 0.0080 0.0089
3: 0.0068 0.0090
4: 0.0062 0.0095
5: 0.0059 0.0095
6: 0.0061 0.0087
7: 0.0059 0.0088

Epoch Train Valid
0: 0.0504 0.0171
1: 0.0145 0.0203
2: 0.0079 0.0088


Epoch Train Valid
0: 0.0401 0.0170
1: 0.0185 0.0108
2: 0.0084 0.0070
3: 0.0077 0.0076
4: 0.0076 0.0072

"""