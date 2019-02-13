from __future__ import print_function
import os
import cv2
import numpy as np
from skimage.io import imread
from skimage.transform import resize
import Global
import csv
import glob
from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.image as mpim

data_path = 'Data/'

image_width = 96
image_height = 96

ratio = 1.435 # width / height


if os.name == 'nt':
    import win32api, win32con


def folder_is_hidden(p):
    if os.name== 'nt':
        attribute = win32api.GetFileAttributes(p)
        return attribute & (win32con.FILE_ATTRIBUTE_HIDDEN | win32con.FILE_ATTRIBUTE_SYSTEM)
    else:
        return p.startswith('.') #linux-osx


csv_array = []

def load_csv():
    thickness_file = open(data_path + '/thickness.csv')
    csv_reader = csv.reader(thickness_file, delimiter=',')
    count = 0
    for row in csv_reader:
        if count == 0:
            count += 1
            continue
        thickness = float(row[8])
        distance = float(row[5])
        csv_array.append([str(row[0]), thickness, distance])
        count += 1

    thickness_file.close()

def find_max_distance():
    max_distance = 0.0
    index = 0
    index_of_max = 0
    for item in csv_array:
        if max_distance < item[2]:
            max_distance = item[2]
            index_of_max = index
        index += 1
    return max_distance, index_of_max


def find_thickness_and_distance(image_name):
    for row in csv_array:
        if str(row[0]) == str(image_name):
            return True, row[1], row[2]
    return False, 0.0, 0.0


def extract_ROI(image, should_log = False):


    #imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(image, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # ret, thresh = cv2.threshold(image, 127, 255, 0)
    # contours, hierarchy = cv2.findContours(thresh, 1, 2)

    index_of_contour = 0
    index_of_max_contour = 0
    max_area = cv2.contourArea(contours[0])
    for contour in contours:
        area = cv2.contourArea(contour)
        if max_area < area:
            max_area = area
            index_of_max_contour = index_of_contour
        index_of_contour += 1

    septum_contour = contours[index_of_max_contour]
    x, y, w, h = cv2.boundingRect(septum_contour)
    extracted_image = image[y:y+h, x:x+w]
    # rect = cv2.minAreaRect(septum_contour)
    # box = cv2.boxPoints(rect)
    # box = np.int0(box)
    # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # cv2.drawContours(image, [box], 0, (0, 0, 255), 2)
    return extracted_image










    # number_of_rows = image.shape[0]
    # number_of_columns = image.shape[1]
    #
    # threshold = 200
    #
    # row_index_up = 0
    # row_index_bottom = 0
    # col_index_left = 0
    # col_index_right = 0
    #
    # row = image[0,:]
    # v = np.sum(row)
    #
    # for row_index in range(number_of_rows):
    #     row = image[row_index,:]
    #     if np.sum(row) > threshold:
    #         row_index_up = row_index
    #         break
    #
    # for row_index in reversed(range(number_of_rows)):
    #     row = image[row_index, :]
    #     if np.sum(row) > threshold:
    #         row_index_bottom = row_index
    #         break
    #
    # for col_index in range(number_of_columns):
    #     col = image[:, col_index]
    #     if np.sum(col) > threshold:
    #         col_index_left = col_index
    #         break
    #
    # for col_index in reversed(range(number_of_columns)):
    #     col = image[:, col_index]
    #     if np.sum(col) > threshold:
    #         col_index_right = col_index
    #         break
    #
    # extracted_image = image[row_index_up:row_index_bottom, col_index_left:col_index_right]
    #
    # if should_log:
    #     print('Sum in X_Up: {0}'.format(np.sum(image[row_index_up, :])))
    #     print('Sum in X_Bottum: {0}'.format(np.sum(image[row_index_bottom, :])))
    #     print('Sum in Y_Left: {0}'.format(np.sum(image[:, col_index_left])))
    #     print('Sum in Y_Right: {0}'.format(np.sum(image[:, col_index_right])))
    #
    #     print('Extracted image dimension: {0}'.format(np.shape(extracted_image)))
    #     print(row_index_up, row_index_bottom, col_index_left, col_index_right)
    #     print('-'*30)
    #
    # return extracted_image

    # aa = cv2.resize(extracted_image, (0,0), fx=4, fy=4)
    #
    # imgplot = plt.imshow(aa)
    # plt.show()



def make_dataset():

    Global.log('Loading Dataset...')

    list_of_image_names = [f for f in os.listdir(data_path + '/images') if not folder_is_hidden(f)]
    index_of_image = 0

    list_of_images = []
    list_of_thicknesses = []
    list_of_distances = []

    for image_name in sorted(list_of_image_names):

        if ".png" not in image_name:
            Global.log('Not image file found. Skipping...', have_line=False)
            continue

        thickness_found, thickness, distance = find_thickness_and_distance(image_name=image_name)
        if thickness_found == False:
            #print('Skipping image {0}'.format(image_name))
            continue

        # image = imread(os.path.join(data_path + '/images', image_name), as_gray=True)
        # image = np.array(image, dtype=np.float32)
        image = cv2.imread(os.path.join(data_path + '/images', image_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #image = cv2.resize(image, (image_width, image_height))
        list_of_images.append(image)
        list_of_thicknesses.append(thickness)
        list_of_distances.append(distance)

        # if index_of_image % 100 == 0:
        #     Global.log('{0} images processed'.format(index_of_image), have_line=False)

        print('Image name {0} is equivalent to image {1} in list'.format(image_name, index_of_image))
        index_of_image += 1

    Global.log('Making dataset done! Using {0} images'.format(index_of_image), have_line=False)
    return list_of_images, list_of_thicknesses, list_of_distances



def get_dataset(test_ratio = 0.2):

    back = np.zeros((10,10))
    front = np.ones((3,3))

    x_offset = 4
    y_offset = 5

    back[x_offset: x_offset+front.shape[0], y_offset: y_offset+front.shape[1]] = front
    print(back)


    load_csv()
    max_distance, index = find_max_distance()
    print(max_distance, index)
    list_of_images, list_of_thicknesses, list_of_distances = make_dataset()

    max_width = 0.0
    max_height = 0.0
    list_of_extracted_images = []
    i = 0
    for image in list_of_images:
        extracted_image = extract_ROI(image, should_log=False)
        distance = list_of_distances[i]
        enlargment_scale = float(max_distance / distance)
        extracted_image = cv2.resize(extracted_image,(0, 0), fx=enlargment_scale, fy=enlargment_scale, interpolation=cv2.INTER_CUBIC)
        list_of_extracted_images.append(extracted_image)
        extracted_image = np.array(extracted_image)
        current_width = extracted_image.shape[0]
        current_height = extracted_image.shape[1]

        if max_width < current_width:
            max_width = current_width

        if max_height < current_height:
            max_height = current_height


    back_width = max_width
    back_height = int(back_width / ratio)

    if back_height < max_height:
        alpha = float(max_height) / float(back_height)
        back_width = int(back_width * alpha)
        back_height = int(back_height * alpha)


    i = 0
    for extracted_scaled_image in list_of_extracted_images:
        scaled_image_array = np.array(extracted_scaled_image, dtype=np.float32)

        current_width = scaled_image_array.shape[0]
        current_height = scaled_image_array.shape[1]

        x_offset = int((back_width - current_width) / 2)
        y_offset = int((back_height - current_height) / 2)

        back_image = np.zeros((back_width, back_height), dtype=np.float32)
        back_image[x_offset:x_offset + extracted_scaled_image.shape[0], y_offset:y_offset + extracted_scaled_image.shape[1]] = extracted_scaled_image
        cv2.imwrite('temp/' + str(i) + '.png', back_image)
        i += 1


    # ndarray_of_images = np.ndarray(shape=(len(list_of_images), image_width, image_height, 1), dtype=np.float32)
    #
    # index = 0
    # for image in list_of_images:
    #     image = np.array(image)
    #     image = image[:,:, np.newaxis]
    #     ndarray_of_images[index] = image
    #     index += 1
    #
    #
    #
    # number_of_train = int(ndarray_of_images.shape[0] * (1-test_ratio))
    #
    # list_of_images_test = ndarray_of_images[number_of_train:ndarray_of_images.shape[0], :, :, :]
    # list_of_thickness_test = list_of_thicknesses[number_of_train:ndarray_of_images.shape[0]]
    #
    #
    # list_of_images_train = ndarray_of_images[0:number_of_train, :, :, :]
    # list_of_thickness_train = list_of_thicknesses[0:number_of_train]
    #
    # Global.log('Splitting dataset done! \nUsing {0} images for train\nUsing {1} images for test'.format(ndarray_of_images.shape[0],list_of_images_test.shape[0]), have_line=False)
    # return list_of_images_train, list_of_thickness_train, list_of_images_test, list_of_thickness_test

get_dataset()