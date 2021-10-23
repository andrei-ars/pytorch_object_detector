import sys
import os
import glob
import json
import xmltodict
from lxml import objectify
from PIL import Image


def xml_obj_to_ann(xml_obj):

    label = xml_obj['name']
    bndbox = xml_obj['bndbox']
    xmin = int(bndbox['xmin'])
    ymin = int(bndbox['ymin'])
    xmax = int(bndbox['xmax'])
    ymax = int(bndbox['ymax'])
    x = (xmax + xmin) / 2
    y = (ymax + ymin) / 2
    w = xmax - xmin
    h = ymax - ymin
    print(x, y, w, h)
    ann = {
        "label": label,
        "coordinates": {"x": x, "y": y, "width": w, "height": h}
    }
    return ann


def convert_to_yolo_format(annotations, im_size):

    #label_to_index_map = {"w": 0, "t": 1, "o": 2, "n": 3, "d": 4}
    label_to_index_map = {"w": 1, "t": 1, "d": 4, "o": 2, "n": 3}
    W, H = im_size

    yolo_coords = []
    for ann in annotations:
        coord = ann["coordinates"]
        x = round(coord['x'] / W, 5)
        y = round(coord['y'] / H, 5)
        w = round(coord['width'] / W, 5)
        h = round(coord['height'] / H, 5)
        label_index = label_to_index_map[ann["label"]]
        str_coord = "{} {:.5f} {:.5f} {:.5f} {:.5f}".format(label_index, x, y, w, h)
        print(str_coord)
        yolo_coords.append(str_coord)

    return yolo_coords


def crop_picture(annotations, im_path):
    
    border = 5

    crops = []
    im = Image.open(im_path)
    W, H = im.size
    for ann in annotations:
        coord = ann["coordinates"]
        x = coord['x']
        y = coord['y']
        w = coord['width']
        h = coord['height']
        left = x - w // 2 - border
        right = x + w // 2 + border
        top = y - h // 2 - border
        bottom = y + h // 2 + border
        im1 = im.crop((left, top, right, bottom))
        crops.append(im1)
    return crops


def save_images_list(images_list, filename):
    with open(filename, "wt") as fp:
        for im_path in images_list:
            fp.write("{}\n".format(im_path))


def convert_xml_to_ann(in_dir, out_dir, yolo_out_dir=None, crop_out_dir=None, mode=""):

    os.system("mkdir -p {}".format(out_dir))
    if yolo_out_dir:
        os.system("mkdir -p {}".format(yolo_out_dir))
    if crop_out_dir:
        os.system("mkdir -p {}".format(crop_out_dir))

    images_list = []
    darknet_data_path = "/data/5_patexia/33_yolo/data/"

    for xml_path in glob.glob(os.path.join(in_dir, "*.xml")):
        with open(xml_path) as fp:
            print("xml:", xml_path)
            xml_string = fp.read()
            #root = objectify.fromstring(xml_string)
            xml = xmltodict.parse(xml_string)
            print(xml)
            annotation = xml['annotation']
            png_path = annotation['path']
            image_filename = annotation['filename']
            
            #image_path = os.path.join(in_dir, image_filename)
            image_path = os.path.join(darknet_data_path, mode, "images", image_filename)
            images_list.append(image_path)

            im_width = int(annotation['size']['width'])
            im_height = int(annotation['size']['height'])
            im_depth = int(annotation['size']['depth'])

            out_annotations = []
            print()
            xml_objs = annotation['object']
            if type(xml_objs) is list:
                print("object is a list")
                #sys.exit()
                for xml_obj in xml_objs:
                    ann = xml_obj_to_ann(xml_obj)
                    out_annotations.append(ann)

            else:
                ann = xml_obj_to_ann(xml_objs)
                out_annotations.append(ann)

            json_data = [{"image": image_filename, "annotations": out_annotations}]
            json_filename = os.path.splitext(image_filename)[0] + ".json"
            json_path = os.path.join(out_dir, json_filename)
            print("json_data:", json_data)
            print("json_path:", json_path)
            with open(json_path, "wt") as outfp:
                json.dump(json_data, outfp)

            if yolo_out_dir:
                yolo_coords = convert_to_yolo_format(out_annotations, (im_width, im_height))
                txt_filename = os.path.splitext(image_filename)[0] + ".txt"
                out_path = os.path.join(yolo_out_dir, txt_filename)
                with open(out_path, "wt") as outfp:
                    for str_coord in yolo_coords:
                        outfp.write("{}\n".format(str_coord))

            if crop_out_dir:
                im_path = os.path.join(in_dir, image_filename)
                crops = crop_picture(out_annotations, im_path)
                for i, crop in enumerate(crops):
                    out_filename = "{}_{}.png".format(os.path.splitext(image_filename)[0], i)
                    out_path = os.path.join(crop_out_dir, out_filename)
                    crop.save(out_path)

    print(images_list)
    list_filename = os.path.join(darknet_data_path, mode, "list.txt")
    save_images_list(images_list, list_filename)


if __name__ == "__main__":

    #convert_xml_to_ann("train/", "json_annotations/train/", "annotations_yolo/train/")
    #convert_xml_to_ann("valid/", "json_annotations/valid/", "annotations_yolo/valid/")

    #dataset_path = "/data/5_patexia/32_object_detection/dataset_abs/"
    #dataset_path = "/data/5_patexia/3_scanned/6_dataset_v2/7_ADS_obj_det/"
    dataset_path = "/data/5_patexia/3_scanned/6_dataset_v2/8_ADS_obj_det_small/"
    xml_subdir = ""
    img_subdir = ""
    json_subdir = "json_annotations"
    yolo_subdir = "annotations_yolo"
    crop_subdir = "crop_bbox"
    for mode in ["train", "valid"]:
        convert_xml_to_ann(
            in_dir=os.path.join(dataset_path, xml_subdir, mode),
            out_dir=os.path.join(dataset_path, json_subdir, mode),
            yolo_out_dir=os.path.join(dataset_path, yolo_subdir, mode),
            crop_out_dir=os.path.join(dataset_path, crop_subdir, mode),
            mode=mode,
            )
