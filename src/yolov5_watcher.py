import numpy as np
import cv2
import pandas as pd
import imutils
# import pickle
import torch

from src.framewatcher import FrameWatcher
from src.util.log_utils import get_default_logger
from models.experimental import attempt_load

logger = get_default_logger()


class YoloV5Watcher(FrameWatcher):

    DEFAULT_COLOR = (225, 175, 35)
    ANGLE_MULT = np.cos(np.pi * 0.44)

    def __init__(self, model_path='../models/best.pt',
                 class_metadata=dict(), 
                 input_size=640,
                 **kwargs):

        super().__init__(**kwargs)
        self.frame_count = 0
        self.class_metadata = class_metadata
        self.input_size = input_size
        logger.info('YoloV5Detector loading {}'.format(model_path))
        self.model = attempt_load(model_path).fuse().autoshape()

    def _custom_processing(self, timestamp, frame):

        # events = self.detection_events.get(self.frame_count, None)
        results = self.model(frame, size=self.input_size)
        events = pd.DataFrame()
        if len(results.xywh) > 0:
            tmp = np.array(results.xywh[0])
            events['cls'] = tmp[:,5].astype(int)
            events['x'] = tmp[:,0] / frame.shape[1]
            events['y'] = tmp[:,1] / frame.shape[0]
            events['w'] = tmp[:,2] / frame.shape[1]
            events['h'] = tmp[:,3] / frame.shape[0]
            events['conf'] = tmp[:,4]

        if events is not None and events.shape[0] > 0:
            for idx, row in events.iterrows():
                class_index = int(row['cls'])
                class_md = self.class_metadata.get(class_index, dict())
                class_label = class_md.get('label', str(class_index))
                confidence = row.get('conf', 0.0)
 
                text = '{}: {:.01f}%'.format(class_label, 100.0*confidence)

                color = class_md.get('color', self.DEFAULT_COLOR)

                w = int(row['w']*frame.shape[1])
                h = int(row['h']*frame.shape[0])
                x = int(row['x']*frame.shape[1] - w/2)
                y = int(row['y']*frame.shape[0] - h/2)

                # cv2.rectangle(frame, (x,y), (x+w, y+h), color, 1)

                text_y = y - 8 if y - 8 > 8 else y + 9
                # text_y = y + 18
                cv2.putText(frame, text, (x, text_y),
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
