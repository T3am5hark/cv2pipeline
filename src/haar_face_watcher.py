import numpy as np
import cv2

from src.framewatcher import FrameWatcher
from src.util.log_utils import get_default_logger

logger = get_default_logger()


class HaarFaceWatcher(FrameWatcher):

    def __init__(self, name = 'HaarFaceWatcher',
                 display_window_name=None,
                 **kwargs):

        #self._cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        fullpath = '/home/pi/jdm/python/envs/cv2pipeline/lib/python3.5/site-packages/cv2/data/haarcascade_frontalface_default.xml'
        self._cascade = cv2.CascadeClassifier(fullpath)
        logger.info('{} loaded frontalface Haar cascade classifier'.format(name))
        logger.info('{}'.format(self._cascade.__class__))
        super().__init__(name=name, display_window_name=display_window_name, **kwargs)

    def _custom_processing(self, timestamp, frame):

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # logger.debug(gray.shape)
        faces = self._cascade.detectMultiScale(gray, 1.1, 4)

        for face in faces:
            x, y, w, h = face
            cv2.rectangle(frame, (x,y), (x+w, y+h), (225, 50, 50), 2)
            cv2.putText(frame, 'Frontal Face', (x+5, y-2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (225, 50, 50), 2)

        return frame
