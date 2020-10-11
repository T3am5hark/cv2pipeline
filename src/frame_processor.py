import cv2
import time
import argparse
import readchar
from datetime import datetime
from threading import Thread

from src.frame_buffer import FrameBuffer
from src.framewatcher import FrameWatcher
from src.mobilenet_watcher import MobileNetWatcher
from src.util.log_utils import get_default_logger, init_logging
from src.util.general import filename_timestamp

logger = get_default_logger()

class FrameProcessor:

    def __init__(self, vflip=False, hflip=False,
                 frame_buffer: FrameBuffer=None,
                 usb_device=0, sleep_time_s=0.0):
        self.vflip = vflip
        self.hflip = hflip
        if frame_buffer is None:
            frame_buffer = FrameBuffer()
        self.buffer = frame_buffer
        self._stopped = True
        self._usb_device = usb_device
        self._video_capture = None
        self._thread = None
        self._frame_count = 0
        self._sleep_time_s = sleep_time_s

    @property
    def is_running(self):
        return not self._stopped

    @property
    def is_stopped(self):
        return self._stopped

    @property
    def frame_count(self):
        return self._frame_count

    @property
    def usb_device(self):
        return self._usb_device

    @property
    def frame_count(self):
        return self._frame_count

    def run(self):

        self._init_video_capture()

        # Test frame
        ret, frame = self._video_capture.read()
        logger.info('Frame resolution = {}'.format(frame.shape))

        self._thread = Thread(target=self._run_capture, args=())
        self._thread.start()
        self._stopped = False

    def stop(self):
        self._stopped = True
        self._thread.join()

    def _init_video_capture(self):
        logger.info('Initializing video capture from {}'.format(self._usb_device))
        self._video_capture = cv2.VideoCapture(self._usb_device)

    def _get_frame(self):
        success, frame = self._video_capture.read()

    def _run_capture(self):
        logger.info('Frame capture thread is running')

        while True:
            if self._stopped:
                return
            success, frame = self._video_capture.read()
            if success:
                if self.vflip and self.hflip:
                    frame = cv2.flip(frame, -1)
                elif self.vflip:
                    frame = cv2.flip(frame, 0)
                elif self.hflip:
                    frame = cv2.flip(frame, 1)

                self.buffer.new_frame(frame)
                self._frame_count += 1

            time.sleep(self._sleep_time_s)


class PicamFrameProcessor(FrameProcessor):

    def _init_video_capture(self):
        pass

    def _run_capture(self):
        pass


def test(display=False, vflip=False, hflip=False, detect=False):
    logger.info('Testing video capture')

    processor = FrameProcessor(vflip=vflip, hflip=hflip)

    # watcher = FrameWatcher(frame_buffer=processor.buffer,
    #                       display_video=display)
    if detect:
        watcher = MobileNetWatcher(frame_buffer=processor.buffer,
                                   display_video=display)
    else:
        watcher = FrameWatcher(frame_buffer=processor.buffer,
                               display_video=display)

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
    args = parser.parse_args()
    display_video = vars(args)['display_video']
    vflip = vars(args)['vflip']
    hflip = vars(args)['hflip']

    logger.info('display_video={}'.format(display_video))
    logger.info('vflip={}'.format(vflip))
    logger.info('hflip={}'.format(hflip))

    test(display=display_video, vflip=vflip, hflip=hflip)
