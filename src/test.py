import cv2
import time
import argparse
import readchar
from datetime import datetime
from threading import Thread

from src.frame_buffer import FrameBuffer
from src.framewatcher import FrameWatcher
from src.frame_processor import FrameProcessor
# from src.mobilenet_watcher import MobileNetWatcher
# from src.mobilenet_tiny_watcher import MobileNetTinyWatcher
from src.motion_watcher import MotionWatcher
from src.haar_face_watcher import HaarFaceWatcher
from src.util.log_utils import get_default_logger, init_logging
from src.util.general import filename_timestamp

logger = get_default_logger()

def test(display=False, vflip=False, hflip=False,
         detect=False, detect_motion=False,
         frame_width=None, frame_height=None):
    logger.info('Testing video capture')

    processor = FrameProcessor(vflip=vflip, hflip=hflip,
                               frame_height=frame_height,
                               frame_width=frame_width)

    watchers = []
    # watcher = FrameWatcher(frame_buffer=processor.buffer,
    #                       display_video=display)
    if detect:
        #watcher = MobileNetWatcher(frame_buffer=processor.buffer,
        #                           display_video=display)
        #watcher = MobileNetTinyWatcher(frame_buffer=processor.buffer,
        #                               display_video=display)
        watcher = HaarFaceWatcher(frame_buffer=processor.buffer,
                                  display_video=display,
                                  scale_factor=0.5,
                                  detection_scaling_factor=1.2,
                                  full_detection_frame=True)
        watchers.append(watcher)

    if detect_motion:
        watcher = MotionWatcher(frame_buffer=processor.buffer,
                                display_video=display,
                                scale_factor=0.35,
                                full_detection_frame=True)
        watchers.append(watcher)

    # watcher = FrameWatcher(frame_buffer=processor.buffer,
    #                        display_video=display)
    # watchers.append(watcher)
    for watcher in watchers:
        logger.info('Starting {}'.format(watcher.name))
        watcher.run()

    processor.run()
    prevtime=datetime.now()

    fps_frames = 600
    last_framecount = 0

    while True:
        if processor.frame_count - last_framecount >= fps_frames:
            current_time=datetime.now()
            delta_s = (current_time-prevtime).total_seconds()
            fps = float(processor.frame_count-last_framecount) / delta_s
            logger.info('{:08d} {:.02f} FPS'.format(processor.frame_count, fps))
            prevtime=current_time
            last_framecount = processor.frame_count

        k = readchar.readkey()

        if k == 'q':
            logger.info('Stopping threads...')
            break

        if k == 'g':
            fname = filename_timestamp() + '.jpeg'
            timestamp, frame = processor.buffer.get_current_frame()
            cv2.imwrite(fname, frame)
            logger.info('Saved frame as {}'.format(fname))

        time.sleep(0.01)

    for watcher in watchers:
        watcher.stop()
    processor.stop()


if __name__ == '__main__':
    init_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument('--display_video', type=bool, default=False,
                        help='(True/False) display video in LXDE window')
    parser.add_argument('--vflip', type=bool, default=False,
                        help='(True/False) vertically flip video')
    parser.add_argument('--hflip', type=bool, default=False,
                        help='(True/False) horizontally flip video')
    parser.add_argument('--detect', type=bool, default=False,
                        help='(True/False) use MobileNet detection')
    parser.add_argument('--detect_motion', type=bool, default=False,
                        help='(True/False) use MotionDetector')
    parser.add_argument('--frame_height', type=int, default=None,
                        help='Frame height (default None)')
    parser.add_argument('--frame_width', type=int, default=None,
                        help='Frame width (default None)')
    args = parser.parse_args()
    display_video = vars(args)['display_video']
    vflip = vars(args)['vflip']
    hflip = vars(args)['hflip']
    detect = vars(args)['detect']
    detect_motion = vars(args)['detect_motion']

    frame_height = vars(args)['frame_height']
    frame_width = vars(args)['frame_width']

    logger.info('display_video={}'.format(display_video))
    logger.info('vflip={}'.format(vflip))
    logger.info('hflip={}'.format(hflip))
    logger.info('frame_height={}'.format(frame_height))
    logger.info('frame_width={}'.format(frame_width))

    test(display=display_video, vflip=vflip, hflip=hflip,
         detect=detect, detect_motion=detect_motion,
         frame_height=frame_height, frame_width=frame_width)
