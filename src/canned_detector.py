import numpy as np
import cv2
import imutils
import pickle

from src.framewatcher import FrameWatcher
from src.util.log_utils import get_default_logger

logger = get_default_logger()


class CannedDetector(FrameWatcher):

    DEFAULT_COLOR = (225, 175, 35)

    def __init__(self, detection_events,
                 class_metadata=dict(), 
                 **kwargs):

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

                cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)

                text_y = y - 15 if y - 15 > 15 else y + 15
                cv2.putText(frame, class_label, (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                cv2.ellipse(frame, ( int(x+w/2), y+h), (int(w/2), int(h/16)), 
                            0, 0, 360, (35, 25, 25), 1)

        self.frame_count += 1
        return frame, events

    @classmethod
    def load_canned_events(cls, fname):
        with open(fname, 'rb') as f:
            detection_events = pickle.load(f)
        return detection_events
