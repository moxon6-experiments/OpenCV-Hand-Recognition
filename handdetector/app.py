import cv2
from .handdetector import HandDetector, NoHandException
from .util import VideoCamera, NoFrameException


class HandDetectorApp:
    def __init__(self):
        self.cam = VideoCamera(1)
        self.hand_detectors = [HandDetector(), HandDetector()]
        self.active_regions = [get_left_region, get_right_region]
        self.running = True

    def run(self):
        while self.running:
            try:
                full_frame = self.cam.get_frame(flip=True)
                self.detect_hands(full_frame)
            except NoFrameException:
                continue
            self.keypress()

    def detect_hands(self, full_frame):
        for hand_detector, get_region in zip(self.hand_detectors, self.active_regions):
            try:
                region = get_region(full_frame)
                hand = hand_detector.get_hand(region)
                hand.draw(region)
            except NoHandException:
                pass
        self.display_window(full_frame)

    def keypress(self):
        key_val = cv2.waitKey(1)
        if key_val < 0:
            return False
        if key_val == 27:
            self.running = False
        else:
            pressed_key = chr(key_val).lower()
            if pressed_key == "r":
                for hd in self.hand_detectors:
                    hd.start_training()

    def display_window(self, full_frame):
        if max([hd.background_training for hd in self.hand_detectors]) > 0:
            frames = max([hd.background_training for hd in self.hand_detectors])
            cv2.putText(full_frame, "Training Background Removal: Frames Remaining: %s" % frames, (30, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 0, 255), 1, cv2.LINE_AA);
        cv2.imshow("Full Frame", full_frame)


# TODO Generalise region getters

def get_left_region(full_frame):
    height, width = full_frame.shape[:2]
    point_1 = (0, 0)
    point_2 = (int(width*0.35), int(0.5*height))
    cv2.rectangle(full_frame, pt1=point_1, pt2=point_2, color=[255, 0, 0], thickness=1, lineType=8)
    active_region = full_frame[point_1[1]:point_2[1], point_1[0]:point_2[0]]
    return active_region


def get_right_region(full_frame):
    height, width = full_frame.shape[:2]
    point_1 = (int(width*0.65), 0)
    point_2 = (width, int(0.5*height))
    cv2.rectangle(full_frame, pt1=point_1, pt2=point_2, color=[255, 0, 0], thickness=1, lineType=8)
    active_region = full_frame[point_1[1]:point_2[1], point_1[0]:point_2[0]]
    return active_region
