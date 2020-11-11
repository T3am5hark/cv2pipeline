import numpy as np
import cv2

from src.framewatcher import FrameWatcher
from src.util.log_utils import get_default_logger

logger = get_default_logger()


class MotionWatcher(FrameWatcher):

    def __init__(self, name = 'MotionWatcher',
                 scale_factor = 0.5,
                 threshold = 0.05,
                 display_window_name=None,
                 full_detection_frame=False,
                 **kwargs):

        super().__init__(name=name, display_window_name=display_window_name, **kwargs)
        self._scale_factor = scale_factor
        self._threshold = threshold
        self._prev_frame = None
        self._full_detection_frame = full_detection_frame
        if not self._full_detection_frame:
            self._text_size = self._text_size / scale_factor

    def _custom_processing(self, timestamp, frame):

        frame_shape = frame.shape
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (int(frame_shape[1]*self._scale_factor),
                                 int(frame_shape[0]*self._scale_factor)))
        gray = cv2.GaussianBlur(gray, (7,7), 0)
        display_gray= gray

        if self._prev_frame is not None:
            #delta = np.abs(gray.astype(int) - self._prev_frame.astype(int))
            #mask = (delta > self._threshold*255)
            delta = cv2.absdiff(self._prev_frame, gray)
            mask = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]

            mask = cv2.dilate(mask, None, iterations=2)
            contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

            if self._full_detection_frame:
                frame = cv2.resize(frame, (int(frame_shape[1]*self._scale_factor),
                                   int(frame_shape[0]*self._scale_factor)))
                # new_mask = cv2.resize(mask, (frame_shape[1], frame_shape[0]))
                #for i in range(0,3):
                for i in [0, 1]:
                    #frame[:,:,i] = frame[:,:,i] - 0.5*np.logical_not(mask)*frame[:,:,i]
                    frame[:,:,i] = np.minimum(frame[:,:,i]+(2-i)*mask*frame[:,:,i], 255)

            else:
                display_gray = gray * mask

        self._prev_frame = gray

        return frame if self._full_detection_frame else display_gray
