import numpy as np
import cv2


def dist(v1, v2):
    return np.linalg.norm(np.array(v1)-v2)


def get_angle(v1, v2):
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    cos_theta = np.dot(v1, v2)
    return np.arccos(cos_theta)*(180/np.pi)


class VideoCamera:
    def __init__(self, camera_id):
        self.cam = cv2.VideoCapture(camera_id)

    def _read(self):
        ret, frame = self.cam.read()
        if ret:
            return frame
        else:
            raise NoFrameException

    def get_frame(self, flip=False):
        frame = self._read()
        if flip:
            frame = cv2.flip(frame, 1)
        return frame


class NoFrameException(Exception):
    def __init__(self):
        Exception.__init__(self, "No frame received from Camera")
