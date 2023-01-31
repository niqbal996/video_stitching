import os
import sys
import numpy as np
# import pandas as pd
import cv2
import csv
# import imutils

from glob import glob
from itertools import product

source_dir = '/mnt/d/raw_datasets/PORTAL_thorvald/plotwise_field_data_PORTAL'

def process_plot(plot_root_dir, cam_groups=[0,1]):
    videos = glob(os.path.join(plot_root_dir, '*.mp4'))
    gps_file = glob(os.path.join(plot_root_dir, '*_gps.csv'))
    csv.register_dialect('myDialect',
                        delimiter=';',
                        skipinitialspace=True,
                        quoting=csv.QUOTE_ALL)
    gps_dict = {}
    gps_dict['GSE'] = []
    entry = {}
    with open(os.path.join(plot_root_dir, gps_file[0]), 'r') as file:
        gps_data = list(csv.reader(file, dialect='myDialect'))
        gps_data_array = np.array(gps_data)
        gps_time_stamps = gps_data_array[1:, 0].astype(np.float16)
    frame_dict = dict()
    frame_dict_counter = 1
    for sample, j in zip([videos[cam_groups[0]], videos[cam_groups[1]]], cam_groups):
        basename = os.path.basename(sample)[:-4]
        frametimes_path = basename + '_frametimes.csv'

        with open(os.path.join(plot_root_dir, frametimes_path), 'r') as file:
            csv_data = list(csv.reader(file, dialect='myDialect'))
            time_data = np.zeros((int(csv_data[-1][0])+1, 3))
            for frame, i in zip(csv_data, range(len(csv_data))):
                time_data[i, :] = [int(frame[0]), float(frame[1]), float(frame[2])]
        frame_dict[frame_dict_counter] = time_data
        frame_dict_counter += 1

    cap1 = cv2.VideoCapture(videos[cam_groups[0]])
    cap1_cam = 'oak1_{}'.format(os.path.basename(videos[cam_groups[0]][-5]))

    cap2 = cv2.VideoCapture(videos[cam_groups[1]])
    cap2_cam = 'oak1_{}'.format(os.path.basename(videos[cam_groups[1]][-5]))

    # Make folders for storing images in the plot folder
    os.makedirs(name=os.path.join(plot_root_dir, cap1_cam), exist_ok=True)
    os.makedirs(name=os.path.join(plot_root_dir, cap2_cam), exist_ok=True)

    print('[INFO] Writing images to \n {}  AND \n {}'.format(os.path.join(plot_root_dir, cap1_cam), 
                                                             os.path.join(plot_root_dir, cap2_cam)))
    # Processing
    for source_frame_number in range(frame_dict[2].shape[0]):
        print('[INFO] Processing frame number {} / {}'.format(source_frame_number, frame_dict[2].shape[0]), end="\r", flush=True)
        ts1 = frame_dict[2][source_frame_number, 2]
        ts2 = frame_dict[1][:, 2]
        gps_time_stamp_index = np.where(abs(gps_time_stamps - ts1) == np.min(abs(gps_time_stamps - ts1)))
        gps_data_at_time = gps_data[gps_time_stamp_index[0][0]]
        # latitude = float(gps_data_at_time[1])
        # longitude = float(gps_data_at_time[2])

        sync_frame_index = np.where(abs(ts2 - ts1) == np.min(abs(ts2 - ts1)))[0][0]

        cap1.set(cv2.CAP_PROP_POS_FRAMES, sync_frame_index - 1)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, source_frame_number - 1)
        status, frame2 = cap2.read()
        if not status:
            continue
        status, frame1 = cap1.read()
        if not status:
            continue
        # print('[INFO] Matching frame {} and {}'.format(sync_frame_index, source_frame_number))

        frame1 = cv2.rotate(frame1, cv2.ROTATE_90_COUNTERCLOCKWISE)
        frame2 = cv2.rotate(frame2, cv2.ROTATE_90_COUNTERCLOCKWISE)

        file_name_1 = os.path.join(plot_root_dir, cap1_cam, '{}_{:04}.png'.format(cap1_cam, source_frame_number))
        file_name_2 = os.path.join(plot_root_dir, cap2_cam, '{}_{:04}.png'.format(cap2_cam, source_frame_number))
        cv2.imwrite(file_name_1, frame1)
        cv2.imwrite(file_name_2, frame2)


        # entry['img_name'] = os.path.basename(file_name_2)
        # entry['img_x'] = global_pos_x
        # entry['img_y'] = global_pos_y
        # entry['gps_lon'] = longitude
        # entry['gps_lat'] = latitude
        # gps_dict['GSE'].append(entry)

        # global_pos_x += 1920
        # entry = {}
        # entry['img_name'] = os.path.basename(file_name_1)
        # entry['img_x'] = global_pos_x
        # entry['img_y'] = global_pos_y
        # entry['gps_lon'] = longitude
        # entry['gps_lat'] = latitude
        # gps_dict['GSE'].append(entry)
        # entry = {}
        # global_pos_y += 1080 - int(1080 * 0.1)
        # (status, stitched) = stitcher.stitch([frame1, frame2])
        # concat = np.hstack((frame1, frame2))
        # concat = cv2.resize(concat, (960, 1080))
        # cv2.imshow('2', concat)
        # cv2.waitKey(0)
    # 
    # import json
    # with open('GCP_valdemar_sample.json', 'w', encoding='utf-8') as f:
    #     json.dump(gps_dict, f, ensure_ascii=False, indent=4)


patches_list = os.walk(source_dir)
for _, plots, _ in patches_list:
    plots = sorted(list(map(int, plots)))
    # for plot in plots[12, 23, 31]:
    for plot in [12, 23, 31]:
        print('[INFO] Processing plot number {}'.format(plot))
        plot_path = os.path.join(source_dir, str(plot))
        videos = glob(os.path.join(plot_path, '*.mp4'))
        process_plot(plot_path, cam_groups=[0,1])
        process_plot(plot_path, cam_groups=[2,3])
        # for video in videos:
        #     cap = cv2.VideoCapture(video)
        #     frame_counter = 0
        #     camera_name = 'oak1_{}'.format(os.path.basename(video)[-5])
        #     os.makedirs(name=os.path.join(plot_path, camera_name),
        #         exist_ok=True)
        #     print('[INFO] Writing images to \n {}'.format(os.path.join(plot_path, camera_name)))
            # while(cap.isOpened()):
            #     # Capture frame-by-frame
            #     ret, frame = cap.read()
            #     if ret == True:
            #         cv2.imwrite(os.path.join(plot_path, 
            #                                 camera_name, 
            #                                 '{:04}.png'.format(frame_counter)),
            #                     frame)
            #         frame_counter += 1
            # cap.release()
 
# Closes all the frames
# cv2.destroyAllWindows()



