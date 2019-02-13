import os
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
from skimage.io import imsave, imread


input_path = '/Users/user/Desktop/Heart/New/Data/mask/yellow'
output_path = '/Users/user/Desktop/Heart/New/Data/mask/septum'

images = os.listdir(input_path)

for image in sorted(images):
    if image.endswith('.png'):
        print(input_path + '/' + image)
        img = imread(input_path + '/' + image, as_gray=True)
        contours, hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        cnt = []
        for contour in contours:
            if len(contour)>15:
                cnt.append(contour)
        pts = np.vstack(cnt)
        hull = cv2.convexHull(pts)
        img = cv2.fillConvexPoly(img, np.array(hull, 'int32'), 255)
        cv2.imwrite(output_path + '/' + image, img)
