import numpy as np
import cv2
from src.detectors.framewatcher import FrameWatcher
from src.util.log_utils import get_default_logger

logger = get_default_logger()

class MobileNetWatcher(FrameWatcher):

    """
    class MobileNetWatcher(FrameWatcher)

    Implements a FrameWatcher processor using a Caffe implementation of a
    MobileNet detector.  
    """

    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

    def __init__(self,
                 path='../models',
                 model='MobileNetSSD_deploy.caffemodel',
                 proto='MobileNetSSD_deploy.prototxt',
                 name = 'MobileNetWatcher',
                 display_window_name=None,
                 confidence_threshold=0.4,
                 ignore_classes=list(),
                 **kwargs):

        self._model_path = path + '/' + model
        self._proto_path = path + '/' + proto
        self._confidence_threshold = confidence_threshold
        logger.info('{} reading model file'.format(name))
        self._net = cv2.dnn.readNetFromCaffe(self._proto_path, self._model_path)
        super().__init__(**kwargs)
        if display_window_name is None:
            self.display_window_name = name
        self.name = name

        self.ignore_classes = ignore_classes

        logger.info('{}'.format(self.display_video))

    def _custom_processing(self, timestamp, frame):
        # Implement MobileNet detection.
        # This implementation works from a fixed 300x300 image size.  

        (h,w) = frame.shape[:2]
        # args: image, scale_factor, shape, mean_factor
        # scalefactor = 0.01
        # ToDo: where did we get this constant from???
        scalefactor = 0.007843
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), scalefactor,
                                     (300, 300), 127.5)
        self._net.setInput(blob)
        detections = self._net.forward()

        events = list()

        for i in np.arange(0, detections.shape[2]):

            confidence = detections[0, 0, i, 2]

            if confidence > self._confidence_threshold:

                idx = int(detections[0, 0, i, 1])
                detected_class = self.CLASSES[idx]

                if detected_class in self.ignore_classes:
                    continue

                events.append(list(detections[0, 0, i, :]))

                # ToDo: Extract annotation code

                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype('int')
                label = '{}: {:.02f}%'.format(detected_class, confidence*100)
                logger.info(label+'\n')
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                              self.COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS[idx], 2)

        return frame, events
