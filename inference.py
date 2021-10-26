#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
First run split_dataset.py to split dataset in train|valid|test parts.
Then run this script.

scp -i ~/.ssh/id_rsa -r andrei@35.224.232.253:/storage/work/cv/obj_det/test/out /ram/tmp/
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
from torch.autograd import Variable

import math
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys
import copy
import glob

import progressbar
from PIL import Image, ImageDraw, ImageFont
from scipy.special import softmax

from nn_models import get_resnet18_classifier
from nn_models import CNN_Net

import data_factory
from data_factory import get_data_transforms
from accuracy import *
from mathfunctions import softmax_probabilities
import settings
model_name = settings.model_name
num_colors = settings.num_colors
image_width = settings.image_width
displayed_size = (500, 500)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


"""
if model_name == "resnet":
    model = get_resnet18_classifier(output_size=5)
elif model_name == "torchvision":
    model = get_torchvision_model(output_size=5)
elif model_name == "custom":
    model = CNN_Net(output_size=5, num_input_channels=1)

model = model.to(device)

model_path = "model_state.pt"
model.load_state_dict(torch.load(model_path))
model.eval()
"""

model = torch.load("model_full.pt")
print("Loading done.")


def inference(model, img, k=1):

    #image_width = 224
    #image_size = (image_width, image_width)
    #img = img.resize(image_size)
    #print("img size:", img.size)
    image_width = img.size[0]

    img = get_data_transforms(image_width)['valid'](img)
    #inputs = Variable(img, volatile=True)
    inputs = Variable(img, requires_grad=False)
    inputs = inputs.view(1, inputs.size(0), inputs.size(1), inputs.size(2))
    inputs = inputs.to(device)

    outputs = model(inputs) # inference
    output = outputs[0].detach().cpu().numpy()
    #max_index = np.argmax(output)
    #max_prob = normalized_output[max_index]
    #topk_predicts = list(output.argsort()[::-1][:k])
    
    return output




# model testing
def model_testing(model, src_dir):

    src_dir = src_dir.rstrip('/')
    subdirs = os.listdir(src_dir)

    res1_list = []
    res6_list = []

    for class_name in subdirs:
        subdir = src_dir + '/' + class_name
        if not os.path.isdir(subdir): continue
        file_names = os.listdir(subdir)
        num_files = len(file_names)
        print('\nclass={}, num_files={}'.format(class_name, num_files))

        for file_name in file_names:
            file_path = subdir + '/' + file_name

            predict, topk_predicts, max_prob = inference(model, file_path, k=6)
            topk_predict_classes = list(map(lambda idx: idx_to_class[idx], topk_predicts))

            predict_class = idx_to_class[predict]

            res1 = 1 if predict_class == class_name else 0
            res6 = 1 if class_name in topk_predict_classes else 0
            res1_list.append(res1)
            res6_list.append(res6)

            print(file_path)
            print('[{}] -- class={}, predict={} (idx={})'.\
                format(res1, class_name, predict_class, predict))
            print('[{}] -- topk:{}'.format(res6, topk_predict_classes))
            #print('[{}] -- topk:{}  ({})'.\
            #    format(res6, topk_classes, topk_predicts))

    return np.mean(res1_list), np.mean(res6_list)


def process_and_show_image(img_path):

    #class_name = '31'
    img = Image.open(img_file)

    if num_colors == 3:
        rgb_img = Image.new("RGB", img.size)
        rgb_img.paste(img)
        img = rgb_img
    
    t1 = time.time()
    #resized_img = rgb_img.resize((image_width, image_width))
    resized_img = img.resize((image_width, image_width))
    
    output = inference(model, resized_img)
    print("output:", output)
    t2 = time.time()
    print("Inference time = {:.2f}".format(t2 - t1))

    img2 = img.resize(displayed_size)
    #bbox = output[0], output[1], 0.1, 0.1
    bbox = output
    draw_bbox(img2, bbox)
    img2.show()


def draw_bbox(img, bbox):
    W, H = img.size
    #print(W, H)
    xc = (bbox[0] + 0.5) * W
    yc = (bbox[1] + 0.5) * H
    w = bbox[2] * W
    h = bbox[3] * W
    x0 = int(xc - w/2)
    y0 = int(yc - h/2)
    x1 = int(xc + w/2) 
    y1 = int(yc + h/2)

    draw = ImageDraw.Draw(img)
    #draw.rectangle((100, 100, 200, 200), None, "#f00", width=3)
    draw.rectangle((x0, y0, x1, y1), None, "#0f0", width=3)
    return img


def draw_grid_and_return_crop(img, output, y_grid):
    S = y_grid
    k = np.argmax(output) # class
    W, H = img.size
    x0 = 5
    x1 = W - 5
    y0 = int(k * H / S)
    y1 = int((k+1) * H / S)

    reserv = (y1 - y0) // 2
    crop_img = img.crop((x0, y0 - reserv, x1, y1 + reserv))

    draw = ImageDraw.Draw(img)
    draw.rectangle((x0, y0, x1, y1), None, "#0f0", width=3)
    
    return crop_img


def extract_crop_and_text(img, nn_output, do_ocr=True, y_grid=30):

    S = y_grid
    k = np.argmax(nn_output) # class
    ps = softmax_probabilities(nn_output)
    confidence = np.max(ps) # or ps[k]
    print("confidence:", confidence)

    W, H = img.size
    x0 = 5
    x1 = W - 5
    y0 = int(k * H / S)
    y1 = int((k+1) * H / S)

    reserv = (y1 - y0) // 2
    crop_img = img.crop((x0, y0 - reserv, x1, y1 + reserv))

    draw = ImageDraw.Draw(img)
    draw.rectangle((x0, y0, x1, y1), None, "#0f0", width=3)
    
    if do_ocr:
        import pytesseract
        custom_config = "--oem 3 --psm 6"
        #text = pytesseract.image_to_string(crop_img, lang='eng', config=custom_config)
        text = pytesseract.image_to_string(img, lang='eng', config=custom_config)

    return {'image':img, 'crop': crop_img, "text": text, "confidence": confidence}


def process_dir(in_dir, out_dir, model_name="custom"):

    #if model_name == "resnet":
    #    num_colors = 3
    #    image_width = 224
    #elif model_name == "custom":
    #    num_colors = 1
    #    image_width = 128

    t1 = time.time()

    for i, img_path in enumerate(glob.glob(os.path.join(in_dir, "*.png"))):
        
        basename = os.path.basename(img_path)
        im_out_path = os.path.join(out_dir, basename)
        text_out_path = os.path.join(out_dir, os.path.splitext(basename)[0] + ".txt")

        img = Image.open(img_path)

        if num_colors == 3:
            rgb_img = Image.new("RGB", img.size)
            rgb_img.paste(img)
            img = rgb_img
            
        input_img = img.resize((image_width, image_width))
        nn_output = inference(model, input_img)
        print("nn output:", nn_output)

        resized_img = img.resize(displayed_size)
        #bbox = output[0], output[1], 0.1, 0.1
        #draw_bbox(img2, bbox=output)
        #crop_img = draw_grid(resized_img, output, y_grid=30)
        #resized_img.save(out_path)

        do_ocr = True
        y_grid = 30
        #crop_img = draw_grid_and_return_crop(img, nn_output, do_ocr=do_ocr, y_grid=y_grid)
        result = extract_crop_and_text(img, nn_output, do_ocr=do_ocr, y_grid=y_grid)
        result['crop'].save(im_out_path)
        #img2.show()
        print("Imaged is saved to {}".format(im_out_path))

        if do_ocr:
            with open(text_out_path, "wt") as fp:
                fp.write("confidence={:4f};\n\n{}\n".format(result['confidence'], result['text']))

    t2 = time.time()
    print("Processed {} images in {:.4f} sec.".format(i+1, t2-t1))

if __name__ == "__main__":

    model_name = settings.model_name
    #process_dir(in_dir="../test/in/", out_dir="../test/out/")
    process_dir(in_dir="../test/valid_in/", out_dir="../test/valid_out/")
    
    
    #process_dir(
    #    in_dir="/data/5_patexia/32_object_detection/test/in/",
    #    out_dir="/data/5_patexia/32_object_detection/test/out/")

    #img_file = '/data/5_patexia/image_classifier/0190_TRNA.png'
    #img_file = '/data/5_patexia/image_classifier/INTV.png'
    #img_file = '/data/tools/dataset/train/0678_JKEASX3RRXEAPX2.png'
    #img_file = '/data/tools/dataset/train/0658_JJUD4UEZRXEAPX0.png'
    #img_file = '/data/tools/dataset/train/0656_JJT4286ZRXEAPX5.png'
    #process_image(img_file)
