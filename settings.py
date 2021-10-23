#export CUDA_VISIBLE_DEVICES=3
import os

#model_name = "custom"
model_name = "resnet"

if model_name == "resnet":
    image_width = 224
    num_colors = 3
    batch_size = 32
elif model_name == "custom":
    image_width = 128
    num_colors = 1
    batch_size = 8

#data_dir = '../dataset/'
#batch_size = 4
#num_workers = 4
#SHOW_BAR = False
#DEBUG = True

