import cv2
from .handdetector import HandDetector


class NoFrameException(Exception):
    def __init__(self):
        Exception.__init__(self, "No frame received from Camera")


def get_active_region(full_frame):
    height, width = full_frame.shape[:2]
    quarter_height = int(height * 0.25)
    half_width = int(width/2)
    point_1 = (half_width-1, quarter_height)
    point_2 = (width, int(3.5*quarter_height))
    cv2.rectangle(full_frame, pt1=point_1, pt2=point_2, color=[255, 0, 0], thickness=1, lineType=8)
    active_region = full_frame[point_1[1]:point_2[1], point_1[0]:point_2[0]]
    return active_region


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


class HandDetectorApp:
    def __init__(self):
        self.hand_detector = HandDetector()
        self.cam = VideoCamera(1)

    def run(self):
        while True:
            try:
                full_frame = self.cam.get_frame(flip=True)
                active_region = get_active_region(full_frame)
                self.draw_hand(active_region)
                self.display_window(full_frame)
                if self.keypress():
                    return
            except NoFrameException:
                continue

    def draw_hand(self, active_region):
        hand = self.hand_detector.get_hand(active_region)
        hand.draw(active_region)

    def keypress(self):
        key_val = cv2.waitKey(1)
        if key_val < 0:
            return False
        if key_val == 27:
            return True
        else:
            pressed_char = chr(key_val).lower()
            self.hand_detector.press_key(pressed_char)

    def display_window(self, full_frame):
        if self.hand_detector.background_training:
            cv2.putText(full_frame, "Training Background Removal: Frames Remaining: %s" % self.hand_detector.background_training, (30, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,0,255), 1, cv2.LINE_AA);
        cv2.imshow("Full Frame", full_frame)
