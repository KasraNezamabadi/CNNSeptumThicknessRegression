import numpy as np
import os
import xlrd
import csv
import sys
#sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2
import matplotlib.pyplot as plt
from math import sqrt
import json


base_path = '/Users/user/Desktop/Heart/New'

gt_frames_path = base_path + '/es_ed'
mask_frames_path = base_path + '/Data/mask/yellow'
matched_gt_path = base_path + '/Data/image'
gt_frames_json_path = base_path+'/clean_frame_json'

masks_frames = os.listdir(mask_frames_path)
gt_frames = os.listdir(gt_frames_path)
jsons = os.listdir(gt_frames_json_path)

count_number = 0
total_number = len(gt_frames)


def sector_mask(shape,centre,radius,angle_range):
    """
    Return a boolean mask for a circular sector. The start/stop angles in
    `angle_range` should be given in clockwise order.
    """

    x,y = np.ogrid[:shape[0],:shape[1]]
    cx,cy = centre
    tmin,tmax = np.deg2rad(angle_range)

    # ensure stop angle > start angle
    if tmax < tmin:
            tmax += 2*np.pi

    # convert cartesian --> polar coordinates
    r2 = (x-cx)*(x-cx) + (y-cy)*(y-cy)
    theta = np.arctan2(x-cx,y-cy) - tmin

    # wrap angles between 0 and 2*pi
    theta %= (2*np.pi)

    # circular mask
    circmask = r2 <= radius*radius

    # angular mask
    anglemask = theta <= (tmax-tmin)

    return circmask*anglemask

def angle_three_points(center, top, down):
    a = np.array(top)
    b = np.array(center)
    c = np.array(down)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    # print np.degrees(angle)
    return np.degrees(angle)

path_to_excel = base_path + '/list_of_thickness.xlsx'

xl_workbook = xlrd.open_workbook(path_to_excel)
sheet = xl_workbook.sheet_by_index(0)



# csv_file = open('thickness.csv', "w")
# writer = csv.writer(csv_file, delimiter=',')
# writer.writerow(['Image_Name', 'Patient_ID', 'Frame_Type', 'Echo_number', 'Scale', 'Distance', 'Radius', 'Starting_Point' 'Thickness']) # 0 1 2 3 4 5 6

# def insert_into_csv(name_of_image, patient_id, frame_type, echo_number, scale, distance, radious, starting_point, thickness):
#     writer.writerow([name_of_image, patient_id, frame_type, echo_number, scale, distance, radious, starting_point, thickness])

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def find_thickness(patient_id, echo_type):
    thickness = 0.0
    for row_index in range(1, sheet.nrows):
        if patient_id == str(int(sheet.cell_value(row_index, 0))):
            row = sheet.row_values(row_index)
            if echo_type == 'ED1':
                if is_number(row[9]):
                    thickness = float(row[9])
            elif echo_type == 'ES1':
                if is_number(row[10]):
                    thickness = float(row[10])
            elif echo_type == 'ED2':
                if is_number(row[11]):
                    thickness = float(row[11])
            elif echo_type == 'ES2':
                if is_number(row[12]):
                    thickness = float(row[12])

    if thickness != 0.0:
        return thickness, True
    else:
        return 0.0, False






count_number = -1
for gt_frame in sorted(gt_frames):
    count_number = count_number +1

    patient_id = gt_frame.split(' ')[0]
    echo_type = (gt_frame.split(' ')[1]).split('.')[0]

    thickness, has_value = find_thickness(patient_id=patient_id, echo_type=echo_type)
    #print(gt_frame + ' - Thickness: ' + str(thickness))


    if gt_frame[:-4]+'.png' in masks_frames:
        figure_read = cv2.imread(gt_frames_path + '/' + gt_frame)


    # parse filename
        folder = gt_frame.split(' ')[0] + ' ' +gt_frame.split(' ')[1][2]

        if folder+'.json' in jsons:
            with open(gt_frames_json_path+'/'+folder+'.json') as json_data:
                data = json.load(json_data)

            # generating the mask for croping based on data
            zero_angle_point = [data['image_shape'][1], data['starting_point'][1]]
            angle1 = angle_three_points(data['starting_point'], zero_angle_point, data['right_point'])
            angle2 = angle_three_points(data['starting_point'], zero_angle_point, data['leftend_point'])
            mask = sector_mask(data['image_shape'], (data['starting_point'][1], data['starting_point'][0]), data['radius'], (angle1, angle2))
            # for each frame croping based on the mask
            figure_read = cv2.cvtColor(figure_read, cv2.COLOR_RGB2GRAY)
            new_figure = cv2.resize(figure_read, (data['image_shape'][1], data['image_shape'][0]))
            new_figure[~mask] = 0
            # Clean the gt frames
            cv2.imwrite(matched_gt_path + '/' + str(count_number) + '.png', new_figure)
            #cv2.imwrite(matched_gt_path + '/' + str(count_number) + '.png', new_figure)

            #clean masks
            mask_read = cv2.imread(mask_frames_path + '/' + gt_frame[:-4]+'.png')
            mask_read = cv2.cvtColor(mask_read, cv2.COLOR_RGB2GRAY)
            new_mask = cv2.resize(mask_read, (data['image_shape'][1], data['image_shape'][0]))
            new_mask[~mask] = 0
            cv2.imwrite(mask_frames_path +'/'+ str(count_number) + '.png', new_mask)

            if has_value:
                insert_into_csv(name_of_image=str(count_number) + '.png',
                                patient_id=patient_id,
                                frame_type=echo_type[0:2],
                                echo_number=echo_type[2],
                                scale=data['scale'][0],
                                distance=data['scale'][1],
                                radious=data['radius'],
                                starting_point=data['starting_point'],
                                thickness=thickness)


        else:
            print (folder)

#csv_file.close()