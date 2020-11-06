import numpy as np
import cv2
import imutils

from src.framewatcher import FrameWatcher
from src.util.log_utils import get_default_logger

logger = get_default_logger()


class MotionWatcher(FrameWatcher):

    def __init__(self, name = 'MotionWatcher',
                 scale_factor = 0.5,
                 threshold = 0.05,
                 display_window_name=None,
                 full_detection_frame=False,
                 min_area=49,
                 memory=0.85,
                 **kwargs):

        super().__init__(name=name, display_window_name=display_window_name, **kwargs)
        self._scale_factor = scale_factor
        self._threshold = threshold
        self._prev_frame = None
        self._full_detection_frame = full_detection_frame
        if not self._full_detection_frame:
            self._text_size = self._text_size / scale_factor
        self._min_area = min_area
        self._memory = memory

    def _custom_processing(self, timestamp, frame):

        frame_shape = frame.shape
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (int(frame_shape[1]*self._scale_factor),
                                 int(frame_shape[0]*self._scale_factor)))
        gray = cv2.GaussianBlur(gray, (11, 11), 0)
        display_gray = gray
        if self._prev_frame is None:
            self._prev_frame = gray.copy()

        # delta = np.abs(gray.astype(int) - self._prev_frame.astype(int))
        delta = cv2.absdiff(gray, self._prev_frame)
        mask = cv2.threshold(delta, 255*self._threshold, 255, cv2.RETR_EXTERNAL)[1]
        mask = cv2.dilate(mask, None, iterations=2)
        # mask = (delta > self._threshold*255)
        contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)

        if self._full_detection_frame:
            #frame = cv2.resize(frame, (int(frame_shape[1]*self._scale_factor),
            #                   int(frame_shape[0]*self._scale_factor)))
            # new_mask = cv2.resize(mask, (frame_shape[1], frame_shape[0]))
            #for i in range(0,3):
            #for i in [0, 1, 2]:
            #    frame[:,:,i] = np.minimum(frame[:,:,i]+0.25*(mask/255)*frame[:,:,i], 255)

            for cnt in contours:
                if cv2.contourArea(cnt) < self._min_area:
                    continue

                (x, y, w, h) = cv2.boundingRect(cnt)
                x = int(x / self._scale_factor)
                y = int(y / self._scale_factor)
                w = int(w / self._scale_factor)
                h = int(h / self._scale_factor)
                cv2.rectangle(frame, (x,y), (x+w, y+h), (225, 175, 35), 2)

        else:
            display_gray = gray * mask

        self._prev_frame = (self._memory*self._prev_frame + (1.0-self._memory)*gray).astype(np.uint8)

        return frame if self._full_detection_frame else display_gray
