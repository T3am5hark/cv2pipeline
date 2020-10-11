import time
import cv2
import numpy as np

from datetime import datetime

from threading import Thread
from src.frame_buffer import FrameBuffer
from src.util.log_utils import get_default_logger

logger = get_default_logger()


class FrameWatcher:

    """
    class FrameWatcher

    Base class for frame processors.  Watcher watches a frame buffer and
    processes all available video frames.
    """

    FPS_FRAMES=20

    def __init__(self, frame_buffer: FrameBuffer,
                 name='WatcherProcess',
                 display_video:bool = False,
                 display_window_name = None):
        self._buffer = frame_buffer
        self._frame_index = 0
        self._thread = None
        self._running = False
        self.display_video = display_video
        if display_window_name is None:
            display_window_name = name
        self.display_window_name = display_window_name
        self.name = name
        self._prev_timestamp = datetime.now()

        self._fps_counter=0
        self._fps_time = datetime.now()
        self._fps = 0.0

    @property
    def frame_index(self):
        return self._frame_index

    @property
    def running(self):
        return self._running

    def run(self):

        self._thread = Thread(target=self._watch, args=())
        self._running = True
        self._thread.start()

    def _watch(self):

        try:

            logger.info('{} running'.format(self.name))
            self._prev_timestamp = datetime.now()
            last_log = datetime.now()

            while self._running:
                while self._frame_index != self._buffer.frame_index:
                    self._frame_index = (self._frame_index + 1) % self._buffer.buffer_len

                    frame_tuple = self._buffer.buffer[self._frame_index]

                    if frame_tuple is None:
                        continue

                    timestamp, frame = frame_tuple
                    processed_frame = self._process_frame(timestamp, frame)
                    self._prev_timestamp = timestamp
                    if self.display_video:
                        self._display_video(processed_frame)

                # If frame buffer exhausted, wait 5ms before checking again
                time.sleep(0.005)

                if (datetime.now()-last_log).total_seconds() > 60:
                    logger.info('{} heartbeat {:08d}'.format(self.name, self._buffer.frame_count))
                    last_log = datetime.now()

        except Exception as ex:
            logger.error('Exception caught in {}'.format(self.name))
            logger.exception(ex)

    def stop(self):
        self._running = False
        self._thread.join()
        logger.info('Stopped {}'.format(self.name))

    def _track_fps(self):
        self._fps_counter += 1
        if self._fps_counter == self.FPS_FRAMES:
            new_time = datetime.now()
            time_s = (new_time - self._fps_time).total_seconds()
            self._fps = self.FPS_FRAMES / time_s
            self._fps_time = new_time
            self._fps_counter = 0
        return self._fps

    def _process_frame(self, timestamp, frame):

        time_delta = (timestamp - self._prev_timestamp).total_seconds()
        fps = self._track_fps()
        text = '{} {:.02f} FPS'.format(self.name, fps)

        processed_frame = frame

        color_index = self._buffer.frame_index+1
        radius = 4
        processed_frame = cv2.circle(processed_frame, (8,12),
                                     radius, (10+10*color_index, 40+30*color_index, 200), 2)

        frame_shape = frame.shape
        origin_shadow = (18, 19)
        origin = (17, 16)
        text_color = (150, 120, 50)

        processed_frame = cv2.putText(processed_frame, text, origin,
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
        #processed_frame = cv2.putText(processed_frame, text, origin,
        #                              cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,255,255), 1)
        processed_frame = self._custom_processing(timestamp, processed_frame)

        return processed_frame

    def _custom_processing(self, timestamp, frame):
        return frame

    def _display_video(self, frame):
        cv2.imshow(self.display_window_name, frame)
        cv2.waitKey(1) # 1ms wait
