import os
import cv2
import numpy as np
import json


base_path = '/Users/user/Desktop/Heart/New'

input_path = base_path + '/gt_frames'
result_path = base_path + '/gt_json'
mask_folder = base_path + '/Data/mask/'

files = os.listdir(input_path)
files.sort()

for each_img in files:
    img_path = input_path + '/' + each_img
    img = cv2.imread(img_path)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_backup = img
    # define range of blue in HSV
    lower_green = np.array([50, 100, 100])
    upper_green = np.array([70, 255, 255])
    # Threshold the HSV image to get only blue colors
    mask_g = cv2.inRange(hsv, lower_green, upper_green)
    # Bitwise-AND mask and original image
    res_g = cv2.bitwise_and(img, img, mask=mask_g)

    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    # Threshold the HSV image to get only blue colors
    mask_r = cv2.inRange(hsv, lower_red, upper_red)
    # Bitwise-AND mask and original image
    res_r = cv2.bitwise_and(img, img, mask=mask_r)

    lower_yellow = np.array([25, 100, 100])
    upper_yellow = np.array([35, 255, 255])
    # Threshold the HSV image to get only blue colors
    mask_y = cv2.inRange(hsv, lower_yellow, upper_yellow)
    # Bitwise-AND mask and original image
    res_y = cv2.bitwise_and(img, img, mask=mask_y)

    cv2.imwrite(mask_folder + 'green/' + each_img, mask_g)
    cv2.imwrite(mask_folder + 'red/' + each_img, mask_r)
    cv2.imwrite(mask_folder + 'yellow/' + each_img, mask_y)

    contours_y, hierarchy = cv2.findContours(mask_y, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

print('Done')