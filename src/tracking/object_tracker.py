import cv2
import numpy as np
from src.tracking.kalman_filter import KalmanFilter

class ObjectTracker:

    def __init__(self, class_metadata=None,
                 distance_threshold=0.025):
        self.detected_objects = list()
        self.class_metadata = class_metadata
        self.distance_threshold = distance_threshold

    def update_detection_events(self, frame, events):
        events['tracked'] = False
        for obj in self.detected_objects:
            obj.update_from_events(frame, events)

        for idx, row in events.iterrows():
            if row['tracked'] == False:
                print('New detection')
                print(row)
                obj = DetectedObject(position=(row['x'], row['y']),
                                     class_index=int(row['cls']),
                                     class_metadata=self.class_metadata,
                                     distance_threshold=self.distance_threshold)
                self.detected_objects.append(obj)

        self.cleanup_objects()

    def cleanup_objects(self):
        pass

class DetectedObject:

    def __init__(self, position, 
                 class_index,
                 initial_detection_event=None,
                 prediction_steps=8,
                 interpolate_frames=28,
                 distance_threshold = 0.025,
                 vert_offset = -0.0,
                 update_on_missing=True,
                 class_metadata=None):

        self.position = position
        self.position_plus_one = position
        self.last_detected_position = position
        self.initial_position = position
        self.kf = KalmanFilter.init_2dtracker(initial_pos=position)
        self.prediction_steps = prediction_steps
        self.interpolate_frames = interpolate_frames
        self.initial_detection_event = initial_detection_event
        self.distance_threshold = distance_threshold
        self._no_detection_counter = 0
        self.update_on_missing = update_on_missing
        self.class_index = class_index
        self.last_detection_event = initial_detection_event
        self.vert_offset = vert_offset
        self.class_metadata = class_metadata
        if class_metadata is not None:
            if 'vert_offset' in class_metadata.get(class_index, []):
                self.vert_offset = class_metadata[class_index]['vert_offset']


    @property
    def height(self):
        return self.last_detection_event['h'] if self.last_detection_event is not None else 10

    @property
    def width(self):
        return self.last_detection_event['w'] if self.last_detection_event is not None else 10

    def distance(self, point1, point2=None):
        if point2 is None:
            point2 = self.position_plus_one

        squared_distance = np.power(point1[0]-point2[0], 2) + np.power(point1[1]-point2[1], 2)
        return np.sqrt(squared_distance)

    def update_from_events(self, frame, events):
        
        self._no_detection_counter += 1
        detected = False

        for idx, row in events.iterrows():
            x = row['x']; y = row['y']

            dist = self.distance( (x,y) )
            # print('me:{} det: {} Distance: {:.04f}'.format(self.class_index, row['cls'], dist))
            if int(row['cls']) == self.class_index and dist < self.distance_threshold:
                # print('Detected {}: {}, {}'.format(self.class_index, (x,y), self.position))
                # row['tracked'] = True
                events.loc[idx, 'tracked'] = True
                self.position, S_k = self.kf.update(np.array((x, y)))
                self._no_detection_counter = 0
                self.last_detection_event = row
                detected = True

                x_update = int(x*frame.shape[1])
                y_update = int((y+self.vert_offset*self.height)*frame.shape[0])
                cv2.circle(frame, (x_update, y_update), 4, (180, 200, 180), 2)

                break

        if detected:
            color = (180, 225, 75)
        else:
            color = (25, 225, 255)
            if self._no_detection_counter < self.interpolate_frames:
                x_k, self.position, P_k, S_k = self.kf.advance_no_observation()
            else:
                # We haven't seen a detection event in a long time, don't know where it went!!
                color = (25, 80, 255)

        x_k = self.kf.x_k
        P_k = self.kf.P_k
        for i in np.arange(0, self.prediction_steps):
            x_k, projected_position, P_k, S_k = self.kf.one_step(x=x_k, P=P_k)
            if i == 0:
                self.position_plus_one = projected_position
            x_prj = int(projected_position[0]*frame.shape[1])
            y_prj = int( (projected_position[1]+self.vert_offset*self.height)*frame.shape[0])
            cv2.circle(frame, (x_prj, y_prj), 1*(i+1), color, 1)
            self.projected_position = projected_position

        x_pos = int(self.position[0]*frame.shape[1])
        y_pos = int((self.position[1]+self.vert_offset*self.height)*frame.shape[0])
        cv2.circle(frame, (x_pos, y_pos), 1, color, 3)

