from datetime import datetime, timedelta
import cv2


class FrameBuffer:

    def __init__(self, n_frames=5):
        self._n_frames = n_frames
        self._frame_index = n_frames-1
        self.buffer = list()

        for i in range(0, self.n_frames):
            self.buffer.append(None)

        self._frame_count = 0

    @property
    def frame_count(self):
        return self._frame_count

    @property
    def frame_index(self):
        return self._frame_index

    def new_frame(self, frame):
        # Place a new frame in the buffer
        idx = (self._frame_index + 1) % self._n_frames
        timestamp = datetime.now()

        self.buffer[idx] = (timestamp, frame)
        self._frame_index = idx

    def get_current_frame(self):
        return self.buffer[self._frame_index]
