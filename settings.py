#export CUDA_VISIBLE_DEVICES=3
import os

#model_name = "custom"
model_name = "resnet"

if model_name == "resnet":
    data_path = "/storage/work/cv/obj_det/ads_dataset/9_ADS_obj_get_generated"
    #data_path = "/storage/work/cv/obj_det/ads_dataset/8_ADS_obj_det_small"
    image_width = 224
    num_colors = 3
    batch_size = 32

elif model_name == "custom":
    #data_path = "/data/5_patexia/3_scanned/6_dataset_v2/9_ADS_obj_get_generated"
    data_path = "/data/5_patexia/3_scanned/6_dataset_v2/8_ADS_obj_det_small"
    image_width = 128
    num_colors = 1
    batch_size = 8

#data_dir = '../dataset/'
#batch_size = 4
#num_workers = 4
#SHOW_BAR = False
#DEBUG = True

