import sys
import os
import glob
import random
import json
#import xmltodict
#from lxml import objectify
from PIL import Image
from PIL import ImageOps, ImageChops


def augument_images(im_indir, bg_indir, im_outdir, json_outdir, repeat_times=1):

    os.system("mkdir -p {}".format(im_outdir))
    os.system("mkdir -p {}".format(json_outdir))
    #os.system("rm -rf {}/*".format(im_outdir))
    #os.system("rm -rf {}/*".format(json_outdir))

    images_list = []

    bg_im_paths = glob.glob(os.path.join(bg_indir, "*.png"))
    crop_im_paths = glob.glob(os.path.join(im_indir, "*.png"))
    print("len(bg_images):", len(bg_im_paths))
    print("len(crop_images):", len(crop_im_paths))
    
    crop_images = []
    for path in crop_im_paths:
        crop_images.append(Image.open(path))
    
    for bg_path in bg_im_paths:
        bg_image = Image.open(bg_path)
        bg_filename = os.path.basename(bg_path)
        W, H = bg_image.size
        print()
        print("bg_path:", bg_path)
        print(W, H)
        for k in range(repeat_times):
            #bg = bg_image.copy()
            max_offset = 100
            x_offset = random.randint(-max_offset, max_offset)
            y_offset = random.randint(-max_offset, max_offset)
            bg = ImageChops.offset(bg_image.copy(), x_offset, y_offset)
            print("offset x={}, y={}".format(x_offset, y_offset))
            
            # Find a crop image that has a compatible size
            compatible = False
            while not compatible:
                crop_im_number = random.randint(0, len(crop_images) - 1)
                crop_im = crop_images[crop_im_number]
                w, h = crop_im.size
                resized_w = int(w * random.uniform(0.95, 1.05))
                resized_h = int(h * random.uniform(0.95, 1.05))
                print("w={}, h={}, resized_w={}, resized_h={}".format(w, h, resized_w, resized_h))
                if resized_w < W and resized_w > 0.75*W:
                    aug_crop_im = crop_im.resize((resized_w, resized_h))
                    compatible = True

            x0 = random.randint(0, W - resized_w)
            y0 = random.randint(0, H - resized_h)
            #x0 = random.randint(max(0, x_offset), W - resized_w + min(0, x_offset) )
            #y0 = random.randint(max(0, y_offset), H - resized_h + min(0, y_offset) )
            print("x0={}, y0={}".format(x0, y0))
            xc = x0 + resized_w // 2
            yc = y0 + resized_h // 2
            bg.paste(aug_crop_im, (x0, y0))
            #bg.paste(crop_im, (x0, y0))
            #bg.show()
            im_out_filename = "{}_{}.png".format(os.path.splitext(bg_filename)[0], k)
            json_out_filename = "{}_{}.json".format(os.path.splitext(bg_filename)[0], k)
            im_out_path = os.path.join(im_outdir, im_out_filename)
            json_out_path = os.path.join(json_outdir, json_out_filename)
            #print(im_out_path)
            #print(json_out_path)
            annotations = [{
                "label": "w", 
                "coordinates": {"x": xc, "y": yc, "width": resized_w, "height": resized_h}
                }]
            json_data = [{"image": im_out_filename, "annotations": annotations}]
            print(json_data)
            bg.save(im_out_path)
            with open(json_out_path, "wt") as outfp:
                json.dump(json_data, outfp)


if __name__ == "__main__":

    #for mode in ["train",]:
    data_path = "/data/5_patexia/3_scanned/6_dataset_v2"
    #data_path = "/storage/work/cv/obj_det/ads_dataset"

    in_dir = "8_ADS_obj_det_small"
    out_dir = "9_ADS_obj_get_generated"
    
    os.system("rm -rf {}".format(os.path.join(data_path, out_dir)))
    os.system("mkdir -p {}".format(os.path.join(data_path, out_dir)))
    os.system("mkdir -p {}".format(os.path.join(data_path, out_dir, "json_annotations")))

    for subdir in ['train', 'valid', 'json_annotations']:
        src_path = os.path.join(data_path, in_dir, subdir)
        dst_path = os.path.join(data_path, out_dir)
        cmd = "cp -r {} {}".format(src_path, dst_path)
        os.system(cmd)
        print(cmd)

    bg_indir = os.path.join(data_path, "4_ADS_classification_marked_by_me/train/0_no_info")
    im_indir = os.path.join(data_path, "8_ADS_obj_det_small/crop_bbox/train")
    im_outdir = os.path.join(data_path, out_dir, "train")
    json_outdir = os.path.join(data_path, out_dir, "json_annotations/train")

    augument_images(im_indir, bg_indir, im_outdir, json_outdir, repeat_times=5)


