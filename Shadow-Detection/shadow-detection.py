import cv2
import numpy as np
import os

def isValidContour(cnt_points, min_area):
    if len(cnt_points) < 50:
        return False
    _, __, w, h = cv2.boundingRect(cnt_points)
    return w*h > min_area

def getBoundingBoxArea(contour):
    _, __, w, h = cv2.boundingRect(contour)
    return w * h

def isInsideRect(rect, another):
    x, y, w, h = rect
    x2, y2, w2, h2 = another
    return x2 >x and y2>y and (x2+w2) < (x+w) and (y2+h2) < (y+h)    

def ordinal(idx: int):
    idx += 1 # Idx starts from 0
    if idx==1: return "1st"
    elif idx==2: return "2nd"
    elif idx==3: return "3rd"
    else: return str(idx)+"th"

COLORS = [(18, 25, 55), (204, 120, 41), (119, 202, 71), (15, 141, 127), (151, 41, 117), (205, 17, 70)]

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--folder", help="Your input folder in input folder", type=str)

args = parser.parse_args()

folder_name = args.folder
if not os.path.exists(f"./intput/{folder_name}"):
    raise Exception("Folder does not exists")
input_folder = f"./input/{folder_name}"
os.makedirs(f"debug/{folder_name}", exist_ok=True)

# Initialization
for file in os.listdir(input_folder):
    file_name, fmt = file.split(".")
    img = cv2.imread(f"{input_folder}/{file}")
    height, width, _ = img.shape
    min_area = 0.5 * height * width
    structure = cv2.getStructuringElement(cv2.MORPH_DILATE, (2,2), anchor=(0,0))
    structure2 = cv2.getStructuringElement(cv2.MORPH_DILATE, (2,2), anchor=(0,1))

    canny = cv2.Canny(img, 10, 30)
    original_canny = canny.copy()
    # Remove all pixels inside button
    contours, arch = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    contours.sort(key=lambda cnt: getBoundingBoxArea(cnt), reverse=True)
    all_contour_img = cv2.drawContours(img.copy(), contours, -1, (255, 255, 0), thickness=1)
    button_bounding_box = cv2.boundingRect(contours[0])
    button_x, button_y, button_w, button_h = button_bounding_box
    if (button_w * button_h) / (height * width) < 0.5:
        print(f"{file_name} is not a valid input")
        continue

    # bounding box is too near
    if (button_w * button_h) / (height * width) > 0.95:
        button_bounding_box = cv2.boundingRect(contours[1])
        button_x, button_y, button_w, button_h = button_bounding_box

    # Remove all contours that inside button
    isInsideButton = lambda contour: isInsideRect(button_bounding_box, cv2.boundingRect(contour))
    noise_contours  = list(filter(isInsideButton, contours[1:]))

    for contour in noise_contours:
        for point in contour:
            canny[point[0][1], point[0][0]] = 0

    # Dilate image to merge pixels near with each other
    dilated_img1 = cv2.dilate(canny, structure)
    dilated_img2 = cv2.dilate(canny, structure2)
    dilated_img = cv2.bitwise_and(dilated_img1, dilated_img2)
    del dilated_img1
    del dilated_img2

    contours, arch = cv2.findContours(dilated_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = list(filter(lambda cnt: isValidContour(cnt, min_area), contours))    

    debug_img = cv2.resize(np.hstack((original_canny, canny, dilated_img)), (0, 0), fx=3, fy=3, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(f"debug/{folder_name}/{file_name}-debug.png", debug_img)
    print("Debug image is located at:", f"debug/{folder_name}/{file_name}-debug.png")

    contours_img = img.copy()
    bounding_box_img = img.copy()
    for cnt_idx in range(len(contours)):
        cv2.drawContours(contours_img, contours, cnt_idx, COLORS[cnt_idx%6][::-1], thickness=1)
        cv2.rectangle(bounding_box_img, cv2.boundingRect(contours[cnt_idx]), COLORS[cnt_idx%6][::-1], thickness=1)
    color_img = cv2.resize(np.hstack((img, all_contour_img, contours_img, bounding_box_img)), (0,0), fx=3, fy=3, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(f"debug/{folder_name}/{file_name}-output.png", color_img)
    print("Output image is located at:", f"debug/{folder_name}/{file_name}-output.png")