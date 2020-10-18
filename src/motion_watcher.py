import numpy as np
import cv2

from src.framewatcher import FrameWatcher
from src.util.log_utils import get_default_logger

logger = get_default_logger()


class MotionWatcher(FrameWatcher):

    def __init__(self, name = 'MotionWatcher',
                 scale_factor = 0.5,
                 threshold=0.2,
                 display_window_name=None,
                 full_detection_frame=False,
                 **kwargs):

        super().__init__(name=name, display_window_name=display_window_name, **kwargs)
        self._scale_factor = scale_factor
        self._threshold = 0.2
        self._prev_frame = None
        self._full_detection_frame = full_detection_frame
        if not self._full_detection_frame:
            self._text_size = self._text_size / scale_factor

    def _custom_processing(self, timestamp, frame):

        frame_shape = frame.shape
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (int(frame_shape[1]*self._scale_factor),
                                 int(frame_shape[0]*self._scale_factor)))
        if self._prev_frame is not None:
            delta = np.abs(gray - prev_frame)
            mask = (delta > 255*self._threshold).astype(int)

            frame = [:,:,0] = 0.5*frame[:,:,0] + 0.5*mask*frame[:,:,0]
            frame = [:,:,1] = 0.5*frame[:,:,1] + 0.5*mask*frame[:,:,1]
            frame = [:,:,2] = 0.5*frame[:,:,2] + 0.5*mask*frame[:,:,2]

            gray = gray * mask

        self._prev_frame = gray

        return frame if self._full_detection_frame else gray
