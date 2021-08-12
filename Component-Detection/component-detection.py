# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, cv2, random, time
random.seed(0)

import torch
from torch import Tensor
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.data.datasets import load_coco_json, register_coco_instances
from detectron2.structures import pairwise_iou, pairwise_ioa, Boxes

# Config argument parser
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--input", help="Image file name in input folder", type=str)

args = parser.parse_args()
if not os.path.exists(f"./input/{args.input}"):
    raise Exception("Folder is not exists")

# Load model and config model parameter
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4 
cfg.OUTPUT_DIR = "faster_rcnn_R_50_FPN_3x"

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # model file path that I trained previously
if not torch.cuda.is_available():
    cfg.MODEL.DEVICE = "cpu"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.45  # set a custom testing threshold
predictor = DefaultPredictor(cfg)
register_coco_instances("web_component_screenshot", {}, "dataset/test/_annotations.json", "./test")

test_coco_json = load_coco_json("dataset/test/_annotations.json", "./test", dataset_name="web_component_screenshot")

test_metadata = MetadataCatalog.get("web_component_screenshot")

def getBoundingBoxArea(contour):
    _, __, w, h = cv2.boundingRect(contour)
    return w * h

def expandCropArea(width, height, x1, y1, x2, y2, expand_ratio):
    w, h = x2-x1, y2-y1
    new_x1, new_y1 = torch.round(x1 - (expand_ratio * w)).to(torch.int32), torch.round(y1 - (expand_ratio * h)).to(torch.int32)
    new_x2, new_y2 = torch.round(x2 + (expand_ratio * w)).to(torch.int32), torch.round(y2 + (expand_ratio * h)).to(torch.int32)
    new_x1 = new_x1 if new_x1>=0 else 0
    new_y1 = new_y1 if new_y1>=0 else 0
    new_x2 = new_x2 if new_x2<width else width
    new_y2 = new_y2 if new_y2<height else height    
    return new_x1, new_y1, new_x2, new_y2

def ordinal(idx: int):
    idx += 1 # Idx starts from 0
    if idx==1: return "1st"
    elif idx==2: return "2nd"
    elif idx==3: return "3rd"
    else: return str(idx)+"th"

# Initialize some variables
file = args.input
tokens = file.split(".")
fmt = tokens[-1]
file_name = ".".join(tokens[:-1])
expand_ratio = 0.05
BUTTON_CLASS = 1

# Prediction
start = time.time()
img = cv2.imread(f"input/{file}")
print(f"{file} readed!")
outputs = predictor(img)
print("Reading and predicting process takes", time.time()-start, "s")
v = Visualizer(img[:,:,::-1], metadata=test_metadata)
draw_instances = v.draw_instance_predictions(outputs["instances"].to("cpu"))
cv2.imwrite(f"./output/{file_name}-output.{fmt}", draw_instances.get_image()[:,:,::-1])
print("Prediction output is located at", f"./output/{file_name}-output.{fmt}")

# Filter button and save output to file
if len(outputs["instances"])>0:
    instances = outputs["instances"]
    out = np.zeros(img.shape)
    pred_classes = instances.get("pred_classes")
    pred_scores = instances.get("scores")

    # We only detect shadow on button so we don't need to care about other types
    button_only_list = []
    for idx, box in enumerate(instances.get("pred_boxes")):
        if pred_classes[idx] == BUTTON_CLASS:
            button_only_list.append(torch.round(box).to(torch.int32))
    button_only_tensor = torch.cat(button_only_list).reshape((-1, 4))
    button_only_tensor = button_only_tensor.to(torch.int32)
    button_only = Boxes(button_only_tensor)
    
    # Computer Intersection over Area(IoA) for each pair of prediction box
    # If IoA=1, both boxs is overlap, we will take box that has higher score
    ioa = pairwise_ioa(button_only, button_only)
    equal_one_indexs = torch.where(ioa==1)
    equal_one_indexs = list(zip(*equal_one_indexs))

    removed_idxs = []
    for (idx1, idx2) in equal_one_indexs:        
        if idx1==idx2:
            continue
        if idx1 in removed_idxs or idx2 in removed_idxs:
            continue
        if pred_scores[idx1] > pred_scores[idx2]:
            removed_idxs.append(idx2)
        else:
            removed_idxs.append(idx1)

    # Expand bounding box that contains button to cover both button and shadow
    valid_idxs = [idx for idx in range(button_only.tensor.shape[0]) if idx not in removed_idxs]
    valid_button = button_only_tensor[valid_idxs]
    button_only_img = img.copy()
    for box in valid_button:
        rect = (box[0], box[1], box[2]-box[0], box[3]-box[1])
        cv2.rectangle(button_only_img, rect, (0, 255, 0), thickness=1)
    cv2.imwrite(f"./output/{file_name}-button.{fmt}", button_only_img)
    print("Image containing only detected buttons is located at:", f"./output/{file_name}-button.{fmt}")

    os.makedirs(f"./output/{file_name}", exist_ok=True)
    height, width, _ = img.shape
    for idx, box in enumerate(valid_button):
        x1, y1, x2, y2 = box
        new_x1, new_y1, new_x2, new_y2 = expandCropArea(width, height, x1, y1, x2, y2, expand_ratio)

        tempor_img = img[new_y1:new_y2+1,new_x1:new_x2+1]
        canny = cv2.Canny(tempor_img, 10, 30)
        contours, arch = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Sort by bounding box area
        contours.sort(key=lambda cnt: getBoundingBoxArea(cnt), reverse=True)
        temp_x, temp_y, temp_w, temp_h = cv2.boundingRect(contours[0])
        ratio = (temp_w*temp_h) / ((new_x2-new_x1)*(new_y2-new_y1))
        if ratio >0.9:
            new_x1, new_y1, new_x2, new_y2 = expandCropArea(width, height, x1, y1, x2, y2, 2*expand_ratio)

        cv2.imwrite(f"./output/{file_name}/{ordinal(idx)}-box.png", img[new_y1:new_y2+1,new_x1:new_x2+1])
        print(f"Output of {ordinal(idx)}: ./output/{file_name}/{ordinal(idx)}-box.png")