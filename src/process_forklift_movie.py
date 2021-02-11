# Script used to de-dup the first forklift movie and collect frame metadata 
# for labeling.  

from datetime import datetime
from time import sleep
from collections import OrderedDict

import cv2
import json
import pandas as pd
import pickle

input_movie = '../movies/trimed_fl.mp4'
write_processed_movie = True
output_fname = 'forklift_deduped.mov'

md_path = '../external/forklift_movie-2020_11_30-yolo/obj_train_data'

# movie_res = (640, 360)
movie_res = (1280, 720)

# Frame skip from source video due to frame duplication
skip_count = 0

# In-loop sleep time
sleep_time = 0.05

# Decompose movie with annotated detection frames for training
save_frames = False
save_loc = '../captures/'

# Rescale video for processing & output
# scale_factor = 0.5
scale_factor = 1.0

if write_processed_movie:
    print('Opening movie writer...')
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    writer = cv2.VideoWriter()
    success = writer.open(output_fname, fourcc, 10.0, movie_res, True)
    print('opened = {}'.format(success))

# cap = cv2.VideoCapture(0) # Capture video from camera
cap = cv2.VideoCapture(input_movie)

# Get the width and height of frame
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)

print('w={}, h={}'.format(width, height))

# Init 
framecount = 0
lastframe_time = datetime.now()
skip_counter = 0
retained_framecount = 0
frame_metadata = OrderedDict()

def save_frame(frame, fname, path=save_loc):
    filename = path + '/' + fname
    cv2.imwrite(filename, frame)
    print('Wrote {}'.format(filename))


def save_metadata(events, fname, path=save_loc):
    filename = path + '/' + fname
    with open(filename, 'w') as f:
        json.dump(events, f)
    print('Write {}'.format(filename))


def save_metadata_and_frame(frame, events, fname_base, path=save_loc):
    fname_frame = fname_base + '.jpeg'
    fname_meta = fname_base + '.json'
    save_frame(frame, fname_frame, path=path)
    save_metadata(events, fname_meta, path=path)


def read_detection_metadata(framecount):
    fname = md_path + '/' + 'frame_{:06d}.txt'.format(framecount)
    df = pd.read_csv(fname, sep=' ', names=['cls', 'x', 'y', 'w', 'h'])
    print('{} {}'.format(fname, df.shape))
    return df


while(cap.isOpened()):
    ret, frame = cap.read()

    skip_counter += 1
    if skip_counter < skip_count:
        continue

    skip_counter = 0

    if ret == True:

        df = read_detection_metadata(framecount)
        framecount += 1

        if df.shape[0] == 0:
            continue

        print('Retain frame {}'.format(framecount))
        frame_metadata[retained_framecount] = df
        retained_framecount += 1

        frame_shape = frame.shape

        if scale_factor < 0.99:
            frame = cv2.resize(frame, (int(scale_factor*frame_shape[1]), 
                                       int(scale_factor*frame_shape[0])))

        frame_shape = frame.shape

        if write_processed_movie:
            writer.write(frame)

        cv2.imshow('frame', frame)
        
        if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
            break

        if sleep_time is not None:
            sleep(sleep_time)

    else:
        break

print('Total frames = {}'.format(framecount))

if save_frames:
    save_metadata(detection_events, 'detection_events.json')

if write_processed_movie:
    writer.release()
    writer = None

# Release everything if job is finished
cap.release()
cv2.destroyAllWindows()

with open('retained_metadata.pkl', 'wb') as f:
    pickle.dump(frame_metadata, f)


