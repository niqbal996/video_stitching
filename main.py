import os
import sys
import numpy as np
import pandas as pd
import cv2
import csv
import imutils

from glob import glob
from itertools import product
def matcher(seq1, seq2):
    min(product(arr1, arr2), key=lambda t: abs(t[0] - t[1]))[0]


root_dir = '/media/niqbal/T7/raw_datasets/PORTAL_thorvald/20220922_NumberOfPlants'
videos = glob(os.path.join(root_dir, '*.mp4'))
csv.register_dialect('myDialect',
                     delimiter=';',
                     skipinitialspace=True,
                     quoting=csv.QUOTE_ALL)
frame_dict = dict()
for sample, j in zip(videos, range(len(videos) - 2)):
    basename = os.path.basename(sample)[:-4]
    frametimes_path = basename + '_frametimes.csv'

    with open(os.path.join(root_dir, frametimes_path), 'r') as file:
        csv_data = list(csv.reader(file, dialect='myDialect'))
        time_data = np.zeros((int(csv_data[-1][0])+1, 3))
        for frame, i in zip(csv_data, range(len(csv_data))):
            time_data[i, :] = [int(frame[0]), float(frame[1]), float(frame[2])]
    frame_dict[j+1] = time_data

for source_frame_number in range(frame_dict[2].shape[0]):
    ts1 = frame_dict[2][source_frame_number, 2]
    ts2 = frame_dict[1][:, 2]
    sync_frame_index = np.where(abs(ts2 - ts1) == np.min(abs(ts2 - ts1)))[0][0]
    #video sequence 1
    cap2 = cv2.VideoCapture(videos[1])
    cap2.set(cv2.CAP_PROP_POS_FRAMES, source_frame_number - 1)
    status, frame2 = cap2.read()

    cap1 = cv2.VideoCapture(videos[0])
    cap1.set(cv2.CAP_PROP_POS_FRAMES, sync_frame_index - 1)
    status, frame1 = cap1.read()
    print('[INFO] Matching frame {} and {}'.format(sync_frame_index, source_frame_number))
    stitcher = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()
    images = []
    frame1 = cv2.rotate(frame1, cv2.ROTATE_90_COUNTERCLOCKWISE)
    frame2 = cv2.rotate(frame2, cv2.ROTATE_90_COUNTERCLOCKWISE)
    (status, stitched) = stitcher.stitch([frame2, frame1])
    concat = np.hstack((frame1, frame2))
    concat = cv2.resize(concat, (960, 1080))
    cv2.imshow('2', concat)
    cv2.waitKey(0)
    if status==0:
        cv2.imshow('stitched', stitched)
        cv2.waitKey()
    print('hold')
