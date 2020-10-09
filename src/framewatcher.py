import time

from threading import Thread
from src.frame_buffer import FrameBuffer
from src.util.log_utils import get_default_logger

logger = get_default_logger()


class FrameWatcher:

    """
    class FrameWatcher

    Base class for frame processors.  Watcher watches a frame buffer and
    processes all available video frames.
    """

    def __init__(self, frame_buffer: FrameBuffer,
                 name='WatcherProcess'):
        self._buffer = buffer
        self._frame_index = 0
        self._thread = None
        self._running = False
        self.name = name

    @property
    def frame_index(self):
        return self._frame_index

    @property
    def running(self):
        return self._running

    def run(self):

        self._thread = Thread(target=self._watch, args=())
        self._thread.start()
        logger.info('{} running'.format(self.name))

    def _watch(self):

        while self._running:
            while self._frame_index != self._buffer.frame_index:
                self._frame_index = (self._frame_index + 1) % self._buffer.buffer_len
                timestamp, frame = self._buffer.buffer[self._frame_index]
            # If frame buffer exhausted, wait 10ms before checking again
            time.sleep(0.010)

    def stop(self):
        self._running = False
        self._thread.join()
        logger.info('Stopped {}'.format(self.name))

    def _process_frame(self, timestamp, frame):
        pass
