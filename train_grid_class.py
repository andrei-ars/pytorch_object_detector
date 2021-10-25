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
import pickle
from nn_models import get_torchvision_model, get_resnet18_classifier
from nn_models import CNN_Net
import data_factory
from data_factory import ObjDetDataset
from loss_function import bboxes_loss
SHOW_BAR = False

import settings
model_name = settings.model_name
num_colors = settings.num_colors
image_width = settings.image_width
batch_size = settings.batch_size # = 32
data_path = settings.data_path
data_parts = ['train', 'valid']

dataset_type = "grid_class"
y_grid = 30

pickle_dataset_path = "dataset_{}_{}_{}.pickle".format(image_width, dataset_type, y_grid)


#start_lr = 0.00128
#num_epochs = 150; start_lr = 0.00256; step_size = 20
#num_epochs = 25; start_lr = 0.00256; step_size = 4
#num_epochs = 25; start_lr = 0.00128; step_size = 5
num_epochs = 20; start_lr = 0.00512; step_size = 5
#num_epochs = 5; start_lr = 0.00512; step_size = 2



if os.path.exists(pickle_dataset_path):
    with open(pickle_dataset_path, 'rb') as fp:
        dataset = pickle.load(fp)
        print("Dataset has been loaded succesfully from {}".format(pickle_dataset_path))
else:
    #data_path = "/data/5_patexia/3_scanned/6_dataset_v2/7_ADS_obj_det"
    #data_path = "/data/5_patexia/3_scanned/6_dataset_v2/9_ADS_obj_get_generated"
    #data_path = "/storage/work/cv/obj_det/ads_dataset/9_ADS_obj_get_generated"

    dataset = {}
    dataset['train'] = ObjDetDataset(
        #images_dir="../dataset/train/", 
        #json_dir="../dataset/json_annotations/train/",
        images_dir=os.path.join(data_path, "train"), 
        json_dir=os.path.join(data_path, "json_annotations/train"),
        data_type="train", 
        num_colors=num_colors,
        image_width=image_width,
        dataset_type=dataset_type,
        y_grid=y_grid
    )
    dataset['valid'] = ObjDetDataset(
        #images_dir="../dataset/valid/", 
        #json_dir="../dataset/json_annotations/valid/",
        images_dir=os.path.join(data_path, "valid"), 
        json_dir=os.path.join(data_path, "json_annotations/valid"),
        data_type="valid", 
        num_colors=num_colors,
        image_width=image_width,
        dataset_type=dataset_type,
        y_grid=y_grid
    )

    if len(dataset['train']) > 0 and len(dataset['valid']) > 0:
        with open(pickle_dataset_path, 'wb') as fp:
            pickle.dump(dataset, fp)
            print("Dataset has been saved in {}".format(pickle_dataset_path))
    else:
        raise Exception("Dataset is empty.")

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
                    #print("labels:", labels) # labels: tensor([1, 2, 2, 2])
                    #print("outputs:", outputs)
                    #print("preds:", preds)
                    loss = criterion(outputs, labels)

                    #print("loss:", loss) # tensor(1.1003, grad_fn=<NllLossBackward>)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # show model prediction for classification
                #print('preds: ', preds)
                #print('labels:', labels.data) # labels.data
                #print('match: ', int(torch.sum(preds == labels.data)))

                running_loss += loss.item() * inputs.size(0)
                if dataset_type == "grid_class":
                    running_corrects += torch.sum(preds == labels.data)

            if SHOW_BAR: 
                bar.finish()

            epoch_loss = running_loss / dataset_sizes[phase]
            if dataset_type == "grid_class":
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if dataset_type == "grid_class":
                print('{} loss: {:.4f}, acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
                history[epoch][phase] = {'loss': epoch_loss, 'acc': epoch_acc}
            else:
                print('{} loss: {:.4f}'.format(phase, epoch_loss))
                history[epoch][phase] = {'loss': epoch_loss}
                
            if phase == 'valid':
                ln_rate = scheduler.get_last_lr()
                history[epoch]['ln_rate'] = ln_rate
                print("ln_rate: {}".format(ln_rate))

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

    if dataset_type == "grid_class":
        print("\nEp: ln_rate | TrainLoss ValLoss | TrainAcc ValAcc")
        for epoch in range(num_epochs):
            ln_rate = history[epoch]['ln_rate']
            train_loss = history[epoch]['train']['loss']
            valid_loss = history[epoch]['valid']['loss']
            train_acc = history[epoch]['train']['acc']
            valid_acc = history[epoch]['valid']['acc']
            print('{:02d}: {} | {:.4f} {:.4f} | {:.3f} {:.3f}'.format(epoch, ln_rate, train_loss, valid_loss, train_acc, valid_acc))
    else:
        print("\nEpoch ln_rate train valid")
        for epoch in range(num_epochs):
            ln_rate = history[epoch]['ln_rate']
            train_loss = history[epoch]['train']['loss']
            valid_loss = history[epoch]['valid']['loss']
            print('{:02d}: {} | {:.4f} {:.4f}'.format(epoch, ln_rate, train_loss, valid_loss))

    return model


if __name__ == "__main__":

    if dataset_type == "grid_class":
        criterion = nn.CrossEntropyLoss()
        output_size = y_grid
    elif dataset_type == "obj_det":
        criterion = bboxes_loss
        output_size = 5
    else:
        raise Exception("Bad dataset_type")

    if model_name == "resnet":
        model = get_resnet18_classifier(output_size=output_size, pretrained=False)
        #model = get_torchvision_model(output_size=output_size, pretrained=True)
    elif model_name == "custom":
        model = CNN_Net(output_size=output_size, num_input_channels=1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Observe that all parameters are being optimized
    #optimizer_ft = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer_ft = optim.SGD(model.parameters(), lr=start_lr, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    #exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=3, gamma=0.1)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=step_size, gamma=0.5)

    model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler,
        num_epochs=num_epochs)

    # save model
    torch.save(model.state_dict(), "model_state.pt")
    torch.save(model, "model_full.pt", _use_new_zipfile_serialization=False)


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


Mobile
Epoch Train Valid
17: 0.0403 0.0198
18: 0.0431 0.0197
19: 0.0366 0.0196

17: 0.0149 0.0039
18: 0.0153 0.0041
19: 0.0157 0.0041


==============

new data

Epoch Train Valid
0: 0.2733 0.1828
1: 0.1991 0.3840
2: 0.1533 0.0404
18: 0.0362 0.0189
19: 0.0342 0.0175

0: 1.3119 0.1615
1: 0.4458 0.5909
18: 0.0647 0.0036
19: 0.0645 0.0036

----------
with augmentation:
custom:
Epoch 9/9
----------
train loss: 0.0398
valid loss: 0.0352
ln_rate: [0.0001]

mobile:

Training complete in 32m 37s

Epoch Train Valid
0: 0.1080 0.0357
1: 0.0334 0.0179
2: 0.0262 0.0238
3: 0.0174 0.0098
4: 0.0168 0.0094


"""