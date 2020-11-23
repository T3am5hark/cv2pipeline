import numpy as np
import cv2
import imutils

from src.framewatcher import FrameWatcher
from src.util.log_utils import get_default_logger

logger = get_default_logger()


class MotionWatcher(FrameWatcher):

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

    def _custom_processing(self, timestamp, frame):

        frame_shape = frame.shape
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #gray = cv2.resize(gray, (int(frame_shape[1]*self._scale_factor),
        #                         int(frame_shape[0]*self._scale_factor)))
        gray = cv2.resize(frame, (int(frame_shape[1]*self._scale_factor),
                                 int(frame_shape[0]*self._scale_factor)))

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
            #frame = cv2.resize(frame, (int(frame_shape[1]*self._scale_factor),
            #                   int(frame_shape[0]*self._scale_factor)))
            # new_mask = cv2.resize(mask, (frame_shape[1], frame_shape[0]))
            #for i in range(0,3):
            #for i in [0, 1, 2]:
            #    frame[:,:,i] = np.minimum(frame[:,:,i]+0.25*(mask/255)*frame[:,:,i], 255)


            events = list()
            for cnt in contours:
                if cv2.contourArea(cnt) < self._min_area:
                    continue

                (x, y, w, h) = cv2.boundingRect(cnt)
                x = int(x / self._scale_factor)
                y = int(y / self._scale_factor)
                w = int(w / self._scale_factor)
                h = int(h / self._scale_factor)
                cv2.rectangle(frame, (x,y), (x+w, y+h), (225, 175, 35), 2)

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

        cv2.imshow('bg_image', delta)
        cv2.waitKey(1)

        return (frame, events) if self._full_detection_frame else (display_gray, events)
