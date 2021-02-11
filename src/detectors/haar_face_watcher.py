import cv2

from src.detectors.framewatcher import FrameWatcher
from src.util.log_utils import get_default_logger

logger = get_default_logger()


class HaarFaceWatcher(FrameWatcher):

    def __init__(self, name='HaarFaceWatcher',
                 scale_factor=0.4,
                 display_window_name=None,
                 full_detection_frame=False,
                 detection_scaling_factor=1.25,
                 **kwargs):

        #self._cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        fullpath = '/home/pi/jdm/envs/cv2pipeline/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_default.xml'
        self._cascade = cv2.CascadeClassifier(fullpath)
        logger.info('{} loaded frontalface Haar cascade classifier'.format(name))
        logger.info('{}'.format(self._cascade.__class__))
        super().__init__(name=name, display_window_name=display_window_name, **kwargs)

        self._scale_factor = scale_factor
        self._full_detection_frame = full_detection_frame
        self._detection_scaling_factor = detection_scaling_factor

    def _custom_processing(self, timestamp, frame):

        frame_shape = frame.shape
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (int(frame_shape[1]*self._scale_factor),
                                 int(frame_shape[0]*self._scale_factor)))

        faces = self._cascade.detectMultiScale(gray, self._detection_scaling_factor, 4)

        for face in faces:
            x, y, w, h = face

            if self._full_detection_frame:
                # Display annotations on original frame
                x = int(x / self._scale_factor)
                y = int(y / self._scale_factor)
                w = int(w / self._scale_factor)
                h = int(h / self._scale_factor)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (225, 50, 50), 2)
                cv2.putText(frame, 'Frontal Face', (x+5, y-2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (225, 50, 50), 2)
            else:
                # Display annotations on detection frame
                cv2.rectangle(gray, (x,y), (x+w, y+h), 25, 2)
                cv2.putText(gray, 'Frontal Face', (x+5, y-2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 25, 2)

        return frame if self._full_detection_frame else gray
