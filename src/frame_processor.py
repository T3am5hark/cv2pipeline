import cv2
import time
from datetime import datetime, timedelta
from threading import Thread

from src.util.log_utils import get_default_logger, init_logging
from src.frame_buffer import FrameBuffer

logger = get_default_logger

class FrameProcessor:

    def __init__(self, vflip=False, hflip=False,
                 buffer: FrameBuffer=None,
                 usb_device=0, sleep_time_s=0.0):
        self.vflip = vflip
        self.hflip = hflip
        if buffer is None:
            buffer = FrameBuffer()
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
        self._thread = Thread(target=self._run_capture, args=())
        self._thread.start()
        self._stopped = False
        logger.info('Frame capture thread is running')

    def stop(self):
        self._stopped = True
        self._thread.join()

    def _init_video_capture(self):
        logger.info('Initializing video capture from {}'.format(self._usb_device))
        self._video_capture = cv2.VideoCapture(self._usb_device)

    def _get_frame(self):
        success, frame = self._video_capture.read()

    def _run_capture(self):
        while True:
            if self.stopped:
                return
        success, frame = self._video_capture.read()
        if (success):
            if self.vflip and self.hflip:
                frame = cv2.flip(frame, -1)
            elif self.vflip:
                frame = cv2.flip(frame, 0)
            elif self.hflip:
                frame = cv2.flip(frame, 1)

            buffer.new_frame(frame)
            self._frame_count += 1

        time.sleep(self._sleep_time_s)


class PicamFrameProcessor(FrameProcessor):

    def _init_video_capture(self):
        pass

    def _run_capture(self):
        pass


def test():
    logger = init_logging()

    processor = FrameProcessor()

    processor.run()
    prevtime=datetime.now()

    while True:
        if processor.frame_count % 1000 == 0:
            current_time=datetime.now()
            delta_s = (current_time-prevtime).seconds
            fps = 1000.0 / delta_s
            logger.info('{:.02f} FPS'.format(fps))


if __name__ == '__main__':
    test()
