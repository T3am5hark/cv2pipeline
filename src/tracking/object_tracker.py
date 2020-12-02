import cv2
import numpy as np
from src.tracking.kalman_filter import KalmanFilter

class ObjectTracker:

    def __init__(self):
        objects = list()

    def update_detection_events(self, events):
        foreach obj in objects:
            pass

class DetectedObject:

    def __init__(self, position, 
                 initial_detection_event=None,
                 prediction_steps=4, 
                 interpolate_frames=10,
                 update_on_missing=True):

        self.position = position
        self.last_detected_position = position
        self.initial_position = position
        self.kf = KalmanFilter.init_2dtracker(initial_pos=position)
        self.prediction_steps = prediction_steps
        self.interpolate_frames = interpolate_frames
        self.initial_detection_event = initial_detection_event

        self._no_detection_counter = 0
        self.update_on_missing = update_on_missing

    def distance(self, point):

        squared_distance = np.power(point[0]-self.position[0], 2) + np.power(point[1]-self.position[1], 2)
        return np.sqrt(squared_distance)

    def update_from_events(events):
        pass

    

    
    
