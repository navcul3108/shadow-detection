import cv2
import numpy as np
import os
import random

def isValidContour(cnt_points, min_area):
    if len(cnt_points) < 50:
        return False
    _, __, w, h = cv2.boundingRect(cnt_points)
    return w*h > min_area

def getBoundingBoxArea(contour):
    _, __, w, h = cv2.boundingRect(contour)
    return w * h  

def ordinal(idx: int):
    idx += 1 # Idx starts from 0
    if idx==1: return "1st"
    elif idx==2: return "2nd"
    elif idx==3: return "3rd"
    else: return str(idx)+"th"

def find2LargestNumberIndices(arr):
    if len(arr)<2:
        raise Exception("Array must contain at least 2 elements")
    elif len(arr)==2:
        return (0, 1) if arr[0]>arr[1] else (1,0)    
    else:
        largest_idx, second_largest_idx = (0, 1) if arr[0]>arr[1] else (1, 0)
        for idx, ele in enumerate(arr):
            if idx <2:
                continue
            if ele > arr[largest_idx]:
                second_largest_idx = largest_idx
                largest_idx = idx
            elif arr[second_largest_idx]<ele and ele < arr[largest_idx]:
                second_largest_idx = idx
        return largest_idx, second_largest_idx

COLORS = [(18, 25, 55), (204, 120, 41), (119, 202, 71), (15, 141, 127), (151, 41, 117), (205, 17, 70)]

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--folder", help="Your input folder in input folder", type=str)

args = parser.parse_args()

folder_name = args.folder
if not os.path.isdir(f"./input/{folder_name}/"):
    raise Exception("Folder does not exists")
input_folder = f"./input/{folder_name}"

if not os.path.exists(f"debug/{folder_name}"):
    os.mkdir(f"debug/{folder_name}")
else:
    for file in os.listdir(f"debug/{folder_name}"):
        os.remove(f"debug/{folder_name}/{file}")

structure1 = cv2.getStructuringElement(cv2.MORPH_DILATE, (2, 2), anchor=(0, 0))
structure2 = cv2.getStructuringElement(cv2.MORPH_DILATE, (2, 2), anchor=(0, 1))

# Loop over each file in the folder
for file in os.listdir(input_folder):
    print("="*100)
    file_name, fmt = file.split(".")
    img = cv2.imread(f"{input_folder}/{file}")
    height, width, _ = img.shape
    image_area = height*width
    min_area = 0.5 * height * width

    canny = cv2.Canny(img, 10, 40)
    original_canny = canny.copy()

    # Remove all pixels inside button
    temp_contours, [hierachy] = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contour_areas = list(map(lambda cnt: getBoundingBoxArea(cnt), temp_contours))
    largest_area_idx, second_largest_area_idx = find2LargestNumberIndices(contour_areas)

    # Find contour of button
    button_contour_idx = None
    if contour_areas[largest_area_idx] / image_area<0.5:
        print(f"{file} is not a valid input, ignore detecting!")
        continue
    elif contour_areas[largest_area_idx] / image_area<0.95:
        button_contour_idx = largest_area_idx
    else:
        # If second largest area contour is not acceptable
        if contour_areas[second_largest_area_idx] / image_area < 0.5:
            print(f"{file} is not a valid input, ignore detecting!")
            continue
        else:
            button_contour_idx = second_largest_area_idx
    
    # Remove all pixel inside button
    all_children_indices = []
    for (idx, (contour, hier)) in enumerate(zip(temp_contours, hierachy)):     
        next_idx, pre_idx, first_child_idx, parent_idx = hier
        # is child
        if parent_idx ==button_contour_idx:
            all_children_indices.append(idx)
        elif parent_idx in all_children_indices:
            all_children_indices.append(idx) 
            for point in contour:
                canny[point[0][1], point[0][0]] = 0      

    # Dilate image to merge pixels near with each other
    dilated_img1 = cv2.dilate(canny, structure1)
    dilated_img2 = cv2.dilate(canny, structure2)
    dilated_img = cv2.bitwise_and(dilated_img1, dilated_img2)

    contours, _ = cv2.findContours(dilated_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    valid_contours = list(filter(lambda cnt: isValidContour(cnt, min_area), contours))    
    if len(valid_contours)==1:
        shadow_x, shadow_y, shadow_w, shadow_h = cv2.boundingRect(valid_contours[0])
        button_x, button_y, button_w, button_h = cv2.boundingRect(temp_contours[button_contour_idx])
    else:
        valid_contours_area = list(map(lambda cnt: getBoundingBoxArea(cnt), valid_contours))
        shadow_idx, button_idx = find2LargestNumberIndices(valid_contours_area)
        shadow_x, shadow_y, shadow_w, shadow_h = cv2.boundingRect(valid_contours[shadow_idx])
        button_x, button_y, button_w, button_h = cv2.boundingRect(valid_contours[button_idx])
    offset_left_x = button_x - shadow_x
    offset_right_x = (shadow_x+shadow_w) - (button_x+button_w)
    offset_top_y = button_y - shadow_y
    offset_bottom_y = (shadow_y+shadow_h) - (button_y+button_h)

    if(offset_left_x<2 and offset_right_x<2 and offset_top_y<2 and offset_bottom_y<2):
        print(f"{file} doesn't contain shadow")
        continue

    print("Offset x:", offset_left_x, offset_right_x)
    print("offset y:", offset_top_y, offset_bottom_y)

    debug_img = cv2.resize(np.hstack((original_canny, canny, dilated_img)), (0, 0), fx=3, fy=3, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(f"debug/{folder_name}/{file_name}-debug.png", debug_img)
    print("Debug image is located at:", f"debug/{folder_name}/{file_name}-debug.png")

    bounding_box_img = img.copy()
    cv2.rectangle(bounding_box_img, (shadow_x, shadow_y, shadow_w, shadow_h), COLORS[0][::-1], thickness=1)
    cv2.rectangle(bounding_box_img, (button_x, button_y, button_w, button_h), COLORS[1][::-1], thickness=1)

    output_img = cv2.resize(np.hstack((img, bounding_box_img)), (0,0), fx=3, fy=3, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(f"debug/{folder_name}/{file_name}-output.png", output_img)
    print("Output image is located at:", f"debug/{folder_name}/{file_name}-output.png")

    