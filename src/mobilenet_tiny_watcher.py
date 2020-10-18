import numpy as np
import tensorflow as tf
import cv2

from src.framewatcher import FrameWatcher
from src.util.log_utils import get_default_logger

logger = get_default_logger()


class MobileNetTinyWatcher(FrameWatcher):

    def __init__(self,
                 model='mnet_training.pb',
                 #model='mnet_fast_inference.pb',
                 name='MobileNetTinyWatcher',
                 display_window_name=None,
                 confidence_threshold=0.4,
                 **kwargs):
        self._model_path = model

        self.name = name
        if display_window_name is None:
            display_window_name = name
        self.display_window_name = display_window_name
        self.confidence_threshold = confidence_threshold

        logger.info('{} reading model file')
        with tf.io.gfile.GFile(self._model_path, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
            self._net = graph_def
            # [n.name for n in tf.get_default_graph().as_graph_def().node]
            logger.info('Graph tensors:')
            # graph = tf.compat.v1.get_default_graph()
            # for op in graph.get_operations():
            #    logger.info(op)
            for n in graph_def.node:
                logger.info(n.name)
        super().__init__(name=name, display_window_name=display_window_name, **kwargs)

        logger.info('display_video = {}'.format(self.display_video))

        self._session = tf.compat.v1.Session()
        self._session.graph.as_default()
        tf.import_graph_def(self._net, name='')

    def _custom_processing(self, timestamp, frame):
        rows = frame.shape[0]
        columns = frame.shape[1]
        resized_frame = cv2.resize(frame, (300, 300))
        # Need to convert CV2's BGR to RGB
        inp = resized_frame[:, :, [2, 1, 0]]

        out = self._session.run([self._session.graph.get_tensor_by_name('num_detections:0'),
                                 self._session.graph.get_tensor_by_name('detection_scores:0'),
                                 self._session.graph.get_tensor_by_name('detection_boxes:0'),
                                 self._session.graph.get_tensor_by_name('detection_classes:0')],
                                feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})
        num_detections = int(out[0][0])
        for i in range(num_detections):
            class_id = int(out[3][0][i])
            confidence = float(out[1][0][i])
            bbox = [float(v) for v in out[2][0][i]]
            if confidence > self.confidence_threshold:
                x = bbox[1] * columns
                y = bbox[0] * rows
                right = bbox[3] * columns
                bottom = bbox[2] * rows
                cv2.rectangle(frame, (int(x), int(y)), (int(right), int(bottom)), (125, 255, 51), thickness=2)
        return frame
