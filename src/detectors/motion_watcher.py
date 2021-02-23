# import numpy as np
import cv2
import imutils

from src.detectors.framewatcher import FrameWatcher
from src.util.log_utils import get_default_logger
from src.visualization.drawing import *

logger = get_default_logger()


class MotionWatcher(FrameWatcher):

    """
    class MotionWatcher(FrameWatcher)

    Implements motion-based ROI detection using frame differencing.
    """

    def __init__(self, name = 'MotionWatcher',
                 scale_factor = 0.5,
                 threshold = 0.08,
                 display_window_name=None,
                 full_detection_frame=False,
                 min_area=49,
                 memory=0.75,
                 gaussian_blur_size=(11, 11),
                 dilation_kernel_size=(5, 5),
                 subtract_motion=False,
                 label_text=False,
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
        self._gaussian_blur_size = gaussian_blur_size
        self._dilation_kernel_size = dilation_kernel_size
        self._subtract_motion = subtract_motion
        self._label_text = label_text

    def _custom_processing(self, timestamp, frame):
        # Implement motion-based ROI detection
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #gray = cv2.resize(gray, (int(frame.shape[1]*self._scale_factor),
        #                         int(frame.shape[0]*self._scale_factor)))
        gray = cv2.resize(frame, (int(frame.shape[1]*self._scale_factor),
                                 int(frame.shape[0]*self._scale_factor)))

        gray = cv2.GaussianBlur(gray, self._gaussian_blur_size, 0)
        display_gray = gray
        if self._prev_frame is None:
            self._prev_frame = gray.copy()

        # delta = np.abs(gray.astype(int) - self._prev_frame.astype(int))
        delta = cv2.absdiff(gray, self._prev_frame)

        delta_flat = cv2.cvtColor(delta, cv2.COLOR_BGR2GRAY)
        #delta_flat = np.sum(delta, axis=2).astype(np.uint8)
        mask = cv2.threshold(delta_flat, 255*self._threshold, 255, cv2.RETR_EXTERNAL)[1]
        kernel = np.ones(self._dilation_kernel_size, dtype=np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        # mask = (delta > self._threshold*255)
        contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)

        if self._full_detection_frame:

            events = list()
            for cnt in contours:
                if cv2.contourArea(cnt) < self._min_area:
                    continue

                (x, y, w, h) = cv2.boundingRect(cnt)
                x = int(x / self._scale_factor)
                y = int(y / self._scale_factor)
                w = int(w / self._scale_factor)
                h = int(h / self._scale_factor)

                label = 'motion ({0:04},{0:04})'.format( int(x+w/2), int(y+h/2)) if self._label_text else ''
                self.annotate(frame, (x,y,x+w+1,y+h+1), label=label)

                event = (x, y, w, h)
                events.append(event)

        else:
            display_gray = gray * mask

        if self._subtract_motion:
            prev_frame_copy = self._prev_frame.copy()
        self._prev_frame = (self._memory*self._prev_frame + (1.0-self._memory)*gray).astype(np.uint8)
        if self._subtract_motion:
            # Preserve background against incorporating moving object
            for cnt in contours:
                if cv2.contourArea(cnt) < self._min_area:
                    continue

                (x, y, w, h) = cv2.boundingRect(cnt)
                s = 5
                self._prev_frame[(x+s):(x+w-s), (y+s):(y+h-s)] = \
                    prev_frame_copy[(x+s):(x+w-s), (y+s):(y+h-s)]

        # ToDo: Make this debug display video output optional
        cv2.imshow('bg_image', delta)
        cv2.waitKey(1)

        return (frame, events) if self._full_detection_frame else (display_gray, events)
