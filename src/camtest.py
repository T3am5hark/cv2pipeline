from datetime import datetime
from time import sleep
from collections import OrderedDict

import cv2
import json

from src.detectors.motion_watcher import MotionWatcher

# MobileNet watcher, else use movement detection
use_mobilenet = False

write_processed_movie = False

if use_mobilenet:
    output_fname = 'Mobilenet-SSD.mov'
else:
    output_fname = 'motion.mov'
# movie_res = (640, 360)
movie_res = (1280, 720)

# Frame skip from source video due to frame duplication??
skip_count = 0

# In-loop sleep time
sleep_time = 0.01

# Decompose movie with annotated detection frames for training
save_frames = False
save_loc = '../captures/'

# Rescale video for processing & output
#scale_factor = 0.5
scale_factor = 1.0

if write_processed_movie:
    print('Opening movie writer...')
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    writer = cv2.VideoWriter()
    success = writer.open(output_fname, fourcc, 10.0, movie_res, True)
    print('opened = {}'.format(success))

cap = cv2.VideoCapture(0) # Capture video from camera
# cap = cv2.VideoCapture('../movies/trimed_fl.mp4')

# Get the width and height of frame
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)

print('w={}, h={}'.format(width, height))

ret, frame = cap.read()
print('Frame size = {}x{}'.format(frame.shape[1], frame.shape[0]))

# Init 
framecount = 0
lastframe_time = datetime.now()
skip_counter = 0

if use_mobilenet:

    from src.detectors.mobilenet_watcher import MobileNetWatcher

    watcher = MobileNetWatcher(frame_buffer=None,
                               display_video=True, 
                               confidence_threshold=0.25,
                               ignore_classes=['bottle', ])
    
    # def __init__(self,
    #              model='MobileNetSSD_deploy.caffemodel',
    #              proto='MobileNetSSD_deploy.prototxt',
    #              name = 'MobileNetWatcher',
    #              display_window_name=None,
    #              confidence_threshold=0.4,
    #              **kwargs)

else:
    watcher = MotionWatcher(frame_buffer=None,
                            display_video=True,
                            scale_factor=0.5,
                            threshold=0.04,
                            full_detection_frame=True,
                            min_area=1600,
                            memory=0.1, 
                            gaussian_blur_size=(11, 11),
                            dilation_kernel_size=(19, 19))

detection_events = OrderedDict()


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


while(cap.isOpened()):
    ret, frame = cap.read()

    skip_counter += 1
    if skip_counter < skip_count:
        continue

    skip_counter = 0

    if ret == True:
        framecount += 1

        frame_shape = frame.shape

        frame = cv2.resize(frame, (int(scale_factor*frame_shape[1]), 
                                   int(scale_factor*frame_shape[0])))

        frame_shape = frame.shape

        now = datetime.now()
        td = (now-lastframe_time).total_seconds()
        lastframe_time = now
        fps_text = '{:2.01f} FPS'.format(1.0/td)
        #cv2.putText(frame, fps_text, (10, 30), 
        #           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 250, 100), 1, cv2.LINE_AA)

        # frame = cv2.flip(frame,0)
        # write the flipped frame

        processed_frame, events = watcher.process_frame(now, frame)

        if save_frames and events is not None and len(events) > 0:
            fname = 'frame_{}.jpeg'.format(framecount)
            save_frame(frame, fname)
            # save_metadata_and_frame(frame, events, fname)
            fname_bb = 'frame_{}.bb.jpeg'.format(framecount)
            save_frame(processed_frame, fname_bb)
            detection_events[framecount] = events

        if write_processed_movie:
            writer.write(processed_frame)

        cv2.imshow('frame', processed_frame)
        
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
