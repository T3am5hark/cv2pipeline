import cv2
import time
import argparse
import readchar
from datetime import datetime
from threading import Thread

from src.frame_buffer import FrameBuffer
from src.util.log_utils import get_default_logger
from src.util.general import filename_timestamp

logger = get_default_logger()

class FrameProcessor:

    def __init__(self, vflip=False, hflip=False,
                 frame_buffer: FrameBuffer=None,
                 usb_device=0, sleep_time_s=0.0,
                 frame_width=None, frame_height=None):
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
        self._frame_width = frame_width
        self._frame_height = frame_height

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
        logger.info('dtype={}'.format(str(frame.dtype)))

        self._thread = Thread(target=self._run_capture, args=())
        self._thread.start()
        self._stopped = False

    def stop(self):
        self._stopped = True
        self._thread.join()

    def _init_video_capture(self):
        logger.info('Initializing video capture from {}'.format(self._usb_device))
        self._video_capture = cv2.VideoCapture(self._usb_device)

        if self._frame_height is not None:
            self._video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT,
                                    self._frame_height)
        if self._frame_width is not None:
            self._video_capture.set(cv2.CAP_PROP_FRAME_WIDTH,
                                    self._frame_width)


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
    """
    Placeholder class in the event that we decide to use the RPi
    camera, has to pull frames differently.
    """
    def _init_video_capture(self):
        pass

    def _run_capture(self):
        pass

