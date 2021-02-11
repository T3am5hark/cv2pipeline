import numpy as np
import cv2
import pickle

from src.detectors.framewatcher import FrameWatcher
from src.util.log_utils import get_default_logger

logger = get_default_logger()


class CannedDetector(FrameWatcher):

    """"
    class CannedDetector(FrameWatcher)

    Implements a frame watcher detection object that uses pre-processed detection metadata
    as a mock detector.  This was originally built from hand-labeled frames to build the object
    tracking algorithm and Kalman filter code (prior to YoloV5 detector being in place).
    """"

    DEFAULT_COLOR = (225, 175, 35)
    ANGLE_MULT = np.cos(np.pi * 0.44)

    def __init__(self, detection_events,
                 class_metadata=dict(), 
                 **kwargs):

        """
        To Do: Document structure of class_metadata, refactor into a class
        """

        super().__init__(**kwargs)
        self.detection_events = detection_events
        self.frame_count = 0
        self.class_metadata = class_metadata

    def _custom_processing(self, timestamp, frame):

        events = self.detection_events.get(self.frame_count, None)

        if events is not None:
            for idx, row in events.iterrows():
                class_index = int(row['cls'])
                class_md = self.class_metadata.get(class_index, dict())
                class_label = class_md.get('label', str(class_index))
                color = class_md.get('color', self.DEFAULT_COLOR)

                w = int(row['w']*frame.shape[1])
                h = int(row['h']*frame.shape[0])
                x = int(row['x']*frame.shape[1] - w/2)
                y = int(row['y']*frame.shape[0] - h/2)

                # cv2.rectangle(frame, (x,y), (x+w, y+h), color, 1)

                # TODO: Factor out annotation code for portability and polymorphism
                # Annotation should be portable & swappable

                text_y = y - 8 if y - 8 > 8 else y + 9
                # text_y = y + 18
                cv2.putText(frame, class_label, (x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                steps=40
                increment=360/steps
                start_angle = (self.frame_count % steps)*360/steps
                end_angle = start_angle + increment

                for i in range(steps):
                    mult = np.cos( 2.0*np.pi*float(i)/steps)
                    mult = mult*mult
                    ring_color = (int(color[0]*mult), int(color[1]*mult), int(color[2]*mult))
                    cv2.ellipse(frame, ( int(x+w/2), y+h), (int(w/2), int(self.ANGLE_MULT*w/2)), 
                                0, start_angle+i*increment, end_angle+i*increment, ring_color, 2)

        self.frame_count += 1
        return frame, events

    @classmethod
    def load_canned_events(cls, fname):
        with open(fname, 'rb') as f:
            detection_events = pickle.load(f)
        return detection_events
