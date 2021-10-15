#export CUDA_VISIBLE_DEVICES=3
import os

#model_name = "custom"
model_name = "resnet"

if model_name == "resnet":
    num_colors = 3
    image_width = 224
elif model_name == "custom":
    num_colors = 1
    image_width = 128

#data_dir = '../dataset/'
#batch_size = 4
#num_workers = 4
#SHOW_BAR = False
#DEBUG = True

