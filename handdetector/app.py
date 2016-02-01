import cv2
from .handdetector import HandDetector


class NoFrameException(Exception):
    def __init__(self):
        Exception.__init__(self, "No frame received from Camera")


def get_right_region(full_frame):
    height, width = full_frame.shape[:2]
    quarter_height = int(height * 0.25)
    point_1 = (int(width*0.65), 0)
    point_2 = (width, int(2*quarter_height))
    cv2.rectangle(full_frame, pt1=point_1, pt2=point_2, color=[255, 0, 0], thickness=1, lineType=8)
    active_region = full_frame[point_1[1]:point_2[1], point_1[0]:point_2[0]]
    return active_region


def get_left_region(full_frame):
    height, width = full_frame.shape[:2]
    quarter_height = int(height * 0.25)
    point_1 = (0, 0)
    point_2 = (int(width*0.35), int(2*quarter_height))
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
        self.left_hand_detector = HandDetector()
        self.right_hand_detector = HandDetector()
        self.cam = VideoCamera(1)

    def run(self):
        while True:
            try:
                full_frame = self.cam.get_frame(flip=True)

                left_region = get_left_region(full_frame)
                right_region = get_right_region(full_frame)

                left_hand = self.left_hand_detector.get_hand(left_region)
                right_hand = self.right_hand_detector.get_hand(right_region)

                left_hand.draw(left_region)
                right_hand.draw(right_region)

                self.display_window(full_frame)

                if self.keypress():
                    return
            except NoFrameException:
                continue

    def keypress(self):
        key_val = cv2.waitKey(1)
        if key_val < 0:
            return False
        if key_val == 27:
            return True
        else:
            pressed_char = chr(key_val).lower()
            self.left_hand_detector.press_key(pressed_char)
            self.right_hand_detector.press_key(pressed_char)

    def display_window(self, full_frame):
        if self.left_hand_detector.background_training:
            cv2.putText(full_frame, "Training Background Removal: Frames Remaining: %s" % self.left_hand_detector.background_training, (30, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 0, 255), 1, cv2.LINE_AA);
        cv2.imshow("Full Frame", full_frame)
